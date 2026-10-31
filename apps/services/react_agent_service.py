import os
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
import httpx
from qdrant_client import QdrantClient, models

from apps.schemas.planner_models import GraphHopArgs, PlannerActionOutput, RewrittenSearchArgs
from apps.services.db_service import _row_to_chunk, AMENDING_RELATIONS, DatabaseService
from apps.services.embedding_service import EmbeddingService, Int8Reranker

PLANNER_SYSTEM_PROMPT = """You are the JSON ReAct Planner for a Vietnamese Legal RAG system.
Evaluate the user query and the current scratchpad pointers. Output ONLY valid JSON adhering to this schema:
{
  "thought": "brief reasoning <= 300 chars in Vietnamese",
  "action": "ABSTAIN" | "tool_rewritten_search" | "tool_aggregation_sweep" | "tool_graph_hop" | "FINISH",
  "abstain_reason": "out_of_scope" | "harmful_or_fictional" | "insufficient_evidence" | null,
  "search_args": {"legal_queries": ["query1", "query2"], "year_from": null, "year_to": null, "legal_field": null} | null,
  "agg_args": {"sub_queries": ["chủ đề 1", "chủ đề 2"], "anchor_keywords": ["từ_khóa_1", "từ_khóa_2"], "year_from": null, "year_to": null} | null,
  "graph_args": {"doc_number": "...", "article_code": null, "direction": "all" | "cites" | "amends"} | null
}

Rules:
1. Choose "ABSTAIN" (with abstain_reason="out_of_scope") for non-legal queries (general knowledge, science, history, coding, weather). Never associate trivia with patent/copyright laws.
2. Choose "tool_aggregation_sweep" whenever the user asks to count ("có bao nhiêu", "tổng số", "mấy điều") or exhaustively list ("liệt kê toàn bộ các điều khoản/văn bản...", "các quy định về...") provisions across documents/years—even if Fast Path missed the aggregation flag. Split multi-part topics into 1-4 atomic Vietnamese sub_queries and specify 1-5 core domain anchor_keywords (e.g. ["thuế", "hóa đơn", "ngân sách"]).
3. Choose "tool_rewritten_search" if query is in a foreign language (translate into 1-3 formal Vietnamese statutory queries) or needs decomposed Vietnamese legal subqueries for a specific case.
4. Choose "tool_graph_hop" if scratchpad reveals a key law and we need related citing/amending laws.
5. Choose "FINISH" if scratchpad headers sufficiently cover the legal issue.
"""


def format_scratchpad(pointers: List[Dict[str, Any]], max_chars: int = 3600) -> str:
    if not pointers:
        return "[]"
    lines = []
    total = 0
    for p in reversed(pointers):
        doc = p.get("doc_number") or ""
        art = p.get("article_code") or ""
        score = round(float(p.get("score") or p.get("rerank_score") or p.get("rrf_score") or 0.0), 3)
        lead = (p.get("title_lead_in") or p.get("breadcrumb") or "")[:100]
        line = f"- [{doc}:{art}] (score={score}) {lead}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    lines.reverse()
    return "\n".join(lines) if lines else "[]"


async def call_ollama_planner_async(
    query: str,
    scratchpad: List[Dict[str, Any]],
    step: int,
    elapsed_s: float,
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 45.0,
) -> PlannerActionOutput:
    load_dotenv(".env")
    ollama_host = host or os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    if not ollama_host.startswith("http"):
        ollama_host = f"http://{ollama_host}"
    model_name = model or os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b-instruct-q4_K_M")

    scratch_text = format_scratchpad(scratchpad)
    user_prompt = (
        f"User Query: {query}\n"
        f"Scratchpad:\n{scratch_text}\n"
        f"Telemetry: step={step}, elapsed={elapsed_s:.1f}s"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.85,
            "num_predict": 180,
        },
    }

    url = f"{ollama_host.rstrip('/')}/api/chat"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        raw_json = resp.json()["message"]["content"]
        return PlannerActionOutput.model_validate_json(raw_json)


def execute_rewritten_search(
    search_args: RewrittenSearchArgs,
    embed_service: EmbeddingService,
    qdrant_client: QdrantClient,
    collection_name: str,
    db_service: DatabaseService,
    top_k: int = 25,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    subqueries = search_args.legal_queries or []
    chunk_bank: Dict[str, Dict[str, Any]] = {}
    matched_subqueries: Dict[str, str] = {}
    new_pointers: List[Dict[str, Any]] = []

    must_conds: List[Any] = []
    if search_args.year_from is not None or search_args.year_to is not None:
        range_args = {}
        if search_args.year_from is not None:
            range_args["gte"] = search_args.year_from
        if search_args.year_to is not None:
            range_args["lte"] = search_args.year_to
        must_conds.append(models.FieldCondition(key="issued_year", range=models.Range(**range_args)))

    qfilter = models.Filter(
        must=must_conds or None,
        must_not=[models.FieldCondition(key="effective_status", match=models.MatchValue(value="expired"))],
    )

    for sub_q in subqueries:
        encoded = embed_service.encode(sub_q)
        dense_vec = encoded["dense"]
        sparse_indices = encoded["sparse"]["indices"]
        sparse_values = encoded["sparse"]["values"]

        prefetch = [
            models.Prefetch(query=dense_vec, using="dense_vector", filter=qfilter, limit=top_k),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse_vector",
                filter=qfilter,
                limit=top_k,
            ),
        ]

        pts = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        ).points

        raw_chunks = []
        for rank_idx, pt in enumerate(pts):
            payload = pt.payload or {}
            raw_chunks.append({
                "chunk_id": str(pt.id),
                "doc_id": payload.get("doc_id") or "",
                "doc_number": payload.get("doc_number") or "",
                "doc_title": payload.get("doc_title") or "",
                "doc_type": payload.get("doc_type") or "",
                "issued_year": payload.get("issued_year"),
                "effective_status": payload.get("effective_status") or "active",
                "legal_field": payload.get("legal_field") or "",
                "article_code": payload.get("article_code") or "",
                "clause_code": payload.get("clause_code") or "ALL",
                "breadcrumb": payload.get("breadcrumb") or "",
                "raw_content": payload.get("raw_content") or "",
                "rrf_score": float(getattr(pt, "score", 0.0) or 0.0),
                "rerank_score": 0.0,
                "rank_order": rank_idx + 1,
            })

        hydrated = db_service.hydrate_chunks(raw_chunks)
        lf_target = (search_args.legal_field or "").strip().lower()
        for c in hydrated:
            cid = c["chunk_id"]
            if lf_target:
                chunk_field = (c.get("legal_field") or "").lower()
                doc_title = (c.get("doc_title") or "").lower()
                if lf_target in chunk_field or lf_target in doc_title:
                    c["rrf_score"] = round(float(c.get("rrf_score", 0.0)) + 0.05, 4)
            if cid not in chunk_bank or float(c.get("rrf_score", 0.0)) > float(chunk_bank[cid].get("rrf_score", 0.0)):
                chunk_bank[cid] = c
            matched_subqueries[cid] = sub_q

        if lf_target:
            hydrated.sort(key=lambda x: float(x.get("rrf_score", 0.0)), reverse=True)

        for c in hydrated[:6]:
            new_pointers.append({
                "chunk_id": c["chunk_id"],
                "doc_number": c.get("doc_number", ""),
                "article_code": c.get("article_code", ""),
                "score": c.get("rrf_score", 0.0),
                "title_lead_in": (c.get("breadcrumb") or c.get("raw_content") or "")[:120],
            })

    return chunk_bank, matched_subqueries, new_pointers


def execute_graph_hop(
    graph_args: GraphHopArgs,
    db_service: DatabaseService,
    limit: int = 10,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    doc_number = graph_args.doc_number.strip().upper()
    article_code = graph_args.article_code.strip() if graph_args.article_code else None
    direction = graph_args.direction or "all"

    conn = db_service._get_conn()
    if conn is None:
        return {}, []

    chunk_bank: Dict[str, Dict[str, Any]] = {}
    pointers: List[Dict[str, Any]] = []

    with conn.cursor() as cur:
        cur.execute("SELECT doc_id::text FROM legal_documents WHERE UPPER(doc_number) = %s LIMIT 1", (doc_number,))
        res = cur.fetchone()
        if not res:
            return {}, []
        doc_id = res[0]

        found_rows = []
        if direction in ("amends", "all"):
            cur.execute(
                """
                SELECT DISTINCT lc.chunk_id::text, lc.raw_content, lc.breadcrumb,
                       ld.doc_id::text, ld.doc_number, ld.doc_title, ld.doc_type,
                       EXTRACT(YEAR FROM ld.issued_date)::int,
                       ld.effective_status, ld.legal_field
                FROM document_relations dr
                JOIN legal_documents ld
                  ON ld.doc_id = CASE
                       WHEN dr.target_doc_id = %s::uuid THEN dr.source_doc_id
                       ELSE dr.target_doc_id
                     END
                JOIN legal_chunks lc ON lc.doc_id = ld.doc_id
                WHERE (dr.target_doc_id = %s::uuid OR dr.source_doc_id = %s::uuid)
                  AND LOWER(dr.relation_type) = ANY(%s)
                  AND ld.effective_status != 'expired'
                LIMIT %s
                """,
                (doc_id, doc_id, doc_id, list(AMENDING_RELATIONS), limit),
            )
            found_rows.extend(cur.fetchall())

        if direction in ("cites", "all") and len(found_rows) < limit:
            rem = limit - len(found_rows)
            cur.execute(
                """
                SELECT DISTINCT lc.chunk_id::text, lc.raw_content, lc.breadcrumb,
                       ld.doc_id::text, ld.doc_number, ld.doc_title, ld.doc_type,
                       EXTRACT(YEAR FROM ld.issued_date)::int,
                       ld.effective_status, ld.legal_field
                FROM provision_references pr
                JOIN legal_chunks lc
                  ON lc.chunk_id = CASE
                       WHEN UPPER(pr.target_doc_number) = %s THEN pr.source_chunk_id
                       ELSE pr.resolved_target_chunk_id
                     END
                JOIN legal_documents ld ON lc.doc_id = ld.doc_id
                WHERE (
                    UPPER(pr.target_doc_number) = %s
                    OR pr.source_chunk_id IN (SELECT chunk_id FROM legal_chunks WHERE doc_id = %s::uuid)
                )
                AND (%s::text IS NULL OR pr.target_article_code = %s)
                AND ld.effective_status != 'expired'
                LIMIT %s
                """,
                (doc_number, doc_number, doc_id, article_code, article_code, rem),
            )
            found_rows.extend(cur.fetchall())

    for row in found_rows:
        c = _row_to_chunk(row)
        c["rrf_score"] = 0.05
        c["rerank_score"] = 0.0
        cid = c["chunk_id"]
        chunk_bank[cid] = c
        pointers.append({
            "chunk_id": cid,
            "doc_number": c.get("doc_number", ""),
            "article_code": c.get("article_code", ""),
            "score": 0.05,
            "title_lead_in": (c.get("breadcrumb") or c.get("raw_content") or "")[:120],
        })

    return chunk_bank, pointers


def score_chunk_bank_candidates(
    chunk_bank: Dict[str, Dict[str, Any]],
    cleaned_vn_query: str,
    matched_subqueries: Dict[str, str],
    reranker: Int8Reranker,
    top_candidates_limit: int = 20,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    if not chunk_bank:
        return [], []

    seen_articles: Dict[Any, Dict[str, Any]] = {}
    for c in chunk_bank.values():
        doc = c.get("doc_number") or ""
        art = c.get("article_code") or ""
        key = (doc, art) if (doc and art) else c.get("chunk_id")
        score = max(float(c.get("rerank_score") or 0.0), float(c.get("rrf_score") or 0.0))
        if key not in seen_articles:
            seen_articles[key] = c
        else:
            old_score = max(
                float(seen_articles[key].get("rerank_score") or 0.0),
                float(seen_articles[key].get("rrf_score") or 0.0),
            )
            if score > old_score:
                seen_articles[key] = c

    pruned = sorted(
        seen_articles.values(),
        key=lambda x: max(float(x.get("rerank_score") or 0.0), float(x.get("rrf_score") or 0.0)),
        reverse=True,
    )[:top_candidates_limit]

    base_pairs = [(cleaned_vn_query, reranker.format_chunk_text(c)) for c in pruned]
    base_logits = reranker.score_pairs(base_pairs)

    sub_pairs = []
    sub_indices = []
    for idx, c in enumerate(pruned):
        sub_q = matched_subqueries.get(c.get("chunk_id", ""))
        if sub_q and sub_q.strip() and sub_q.strip() != cleaned_vn_query.strip():
            sub_pairs.append((sub_q, reranker.format_chunk_text(c)))
            sub_indices.append(idx)

    max_logits = list(base_logits)
    if sub_pairs:
        sub_logits = reranker.score_pairs(sub_pairs)
        for orig_idx, sub_logit in zip(sub_indices, sub_logits):
            if sub_logit > max_logits[orig_idx]:
                max_logits[orig_idx] = sub_logit

    return pruned, max_logits

