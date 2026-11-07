"""Re-index gate. Runs on a one-shot Colab T4; started by scripts/embed_gate_launch.py, which fills GATE_INPUT.

Embeds a chunk sample two ways and ranks the dev queries in memory. Nothing is written to Qdrant.
  old = chunk text, as legal_chunks_v1 was built
  new = document title + breadcrumb + chunk text
"""
import base64
import gzip
import json
import os
import subprocess
import sys
import time

GATE_INPUT = "__GATE_INPUT__"
ENCODE_BATCH = 32
MAX_TOKENS = 1024
STARTED = time.time()


def log(message):
    print(f"[{time.time() - STARTED:6.0f}s] {message}", flush=True)


def setup():
    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    log(gpus.strip())
    if "T4" not in gpus:
        sys.exit("ABORT: not a T4")  # an unrecognised --gpu falls back to an A100
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "FlagEmbedding", "psycopg2-binary"], check=True)
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=["cuda:0"])
    log("model ready")
    return model


def connect_postgres():
    import psycopg2

    connection = psycopg2.connect(os.environ["DATABASE_DSN"])
    connection.set_session(readonly=True)
    return connection


def contextual_text(title, breadcrumb, raw_content):
    return f"{title}\n{breadcrumb}\n{raw_content}"


def encode(model, texts):
    return model.encode(texts, batch_size=ENCODE_BATCH, max_length=MAX_TOKENS, return_dense=True, return_sparse=True, return_colbert_vecs=False)


def sparse_matrix(weights, vocabulary_size=250002):
    from scipy.sparse import csr_matrix

    rows, columns, values = [], [], []
    for row, entry in enumerate(weights):
        for token, weight in entry.items():
            rows.append(row)
            columns.append(int(token))
            values.append(float(weight))
    return csr_matrix((values, (rows, columns)), shape=(len(weights), vocabulary_size))


def rrf_top(dense_scores, sparse_scores, depth=100, k=60):
    """Top chunk indices for one query, fusing both rankings the way Qdrant's RRF does."""
    import numpy as np

    fused = {}
    for scores in (dense_scores, sparse_scores):
        for rank, index in enumerate(np.argsort(-scores)[:depth]):
            fused[int(index)] = fused.get(int(index), 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def rank_queries(chunks, chunk_vectors, query_vectors):
    """Returns, per query, the ordered unique (doc, article) list."""
    dense = query_vectors["dense_vecs"] @ chunk_vectors["dense_vecs"].T
    sparse = (sparse_matrix(query_vectors["lexical_weights"]) @ sparse_matrix(chunk_vectors["lexical_weights"]).T).toarray()
    ranked = []
    for dense_row, sparse_row in zip(dense, sparse):
        seen, articles = set(), []
        for index in rrf_top(dense_row, sparse_row):
            key = (chunks[index]["doc_number"], chunks[index]["article_code"])
            if key not in seen:
                seen.add(key)
                articles.append(key)
        ranked.append(articles)
    return ranked


def run_gate(model):
    import re

    payload = json.loads(gzip.decompress(base64.b64decode(GATE_INPUT)))
    connection = connect_postgres()
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT ld.doc_number, ld.doc_title, lc.breadcrumb, lc.raw_content
            FROM legal_chunks lc JOIN legal_documents ld ON ld.doc_id = lc.doc_id
            WHERE ld.doc_number = ANY(%s) OR md5(ld.doc_number) < '18'
            """,
            (payload["priority_docs"],),
        )
        rows = cursor.fetchall()
    log(f"{len(rows)} chunks loaded")

    article_pattern = re.compile(r"Điều\s+\d+[a-zA-ZđĐ]?")
    chunks = []
    for doc_number, title, breadcrumb, raw_content in rows:
        match = article_pattern.search(breadcrumb or "") or article_pattern.search(raw_content or "")
        chunks.append({
            "doc_number": doc_number,
            "article_code": " ".join(match.group(0).split()) if match else "",
            "old": raw_content,
            "new": contextual_text(title, breadcrumb, raw_content),
        })

    query_vectors = encode(model, [q["query"] for q in payload["queries"]])
    results = {}
    for name in ("old", "new"):
        started = time.time()
        vectors = encode(model, [c[name] for c in chunks])
        seconds = time.time() - started
        log(f"{name}: {len(chunks) / seconds:.1f} chunks/s")
        results[name] = rank_queries(chunks, vectors, query_vectors)

    rows_out = []
    for index, query in enumerate(payload["queries"]):
        gold = {(t["doc_number"], t.get("article_code")) for t in query["expected_targets"]}
        row = {"id": query["query_id"], "set": query["set"], "type": query["type"]}
        for name in ("old", "new"):
            ranks = [r for r, key in enumerate(results[name][index], 1) if key in gold]
            row[name] = ranks[0] if ranks else None
        rows_out.append(row)
    print("@@GATE " + json.dumps(rows_out), flush=True)


model = setup()
run_gate(model)
print("@@END", flush=True)
