"""Exports the seed material for the silver benchmark. Read-only on Postgres, no LLM.

Writes one JSON file, so the Colab generation job needs no database credentials:
  articles   stratified single-article seeds for Set A
  pairs      resolved cross-document references for Set M
  temporal   Set T items, gold known from title-sourced `replaces` relations
  structural Set S items, gold counted from the chunks themselves
"""
import os
import sys

os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import argparse
import collections
import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from dotenv import load_dotenv

RE_ARTICLE = re.compile(r"Điều\s+(\d+[a-zA-ZđĐ]?)\b")
RE_YEAR = re.compile(r"/((?:19|20)\d{2})/")
# Articles with no question worth asking: entry into force, execution duties, amendments of other texts.
RE_BOILERPLATE = re.compile(
    r"hiệu lực|thi hành|tổ chức thực hiện|trách nhiệm thực hiện|chuyển tiếp|bãi bỏ|thay thế|"
    r"sửa đổi|bổ sung|đính chính|phạm vi điều chỉnh|đối tượng áp dụng|áp dụng",
    re.IGNORECASE,
)
RE_AMENDING_TITLE = re.compile(r"sửa đổi|bổ sung|bãi bỏ", re.IGNORECASE)
TYPE_QUOTA = {"Luật": 0.25, "Nghị định": 0.30, "Thông tư": 0.45}
ERAS: List[Tuple[str, int, int]] = [("pre2010", 0, 2009), ("2010_17", 2010, 2017), ("2018_22", 2018, 2022), ("2023on", 2023, 9999)]
MIN_ARTICLE_TOKENS, MAX_ARTICLE_TOKENS = 80, 900


def era_of(doc_number: str) -> str:
    match = RE_YEAR.search(doc_number or "")
    year = int(match.group(1)) if match else 0
    return next(name for name, low, high in ERAS if low <= year <= high)


def article_of(breadcrumb: str) -> Optional[Tuple[str, str]]:
    """Returns ("Điều 5", "Điều 5. Title") from the breadcrumb segment naming the article."""
    for part in (breadcrumb or "").split(" > "):
        match = RE_ARTICLE.match(part.strip())
        if match:
            return f"Điều {match.group(1)}", part.strip()
    return None


def body_of(raw_content: str, breadcrumb: str) -> str:
    """Drops the breadcrumb line that split-clause chunks repeat at the top of their text."""
    lines = (raw_content or "").split("\n")
    if lines and " > " in lines[0] and lines[0].strip() in (breadcrumb or ""):
        lines = lines[1:]
    return "\n".join(lines).strip()


def load_articles(cur) -> Dict[Tuple[str, str], Dict[str, Any]]:
    cur.execute(
        """
        SELECT ld.doc_number, ld.doc_title, ld.doc_type, ld.legal_field, lc.chunk_id::text,
               lc.breadcrumb, lc.raw_content, lc.token_count
        FROM legal_chunks lc JOIN legal_documents ld ON ld.doc_id = lc.doc_id
        ORDER BY ld.doc_number, lc.breadcrumb
        """
    )
    articles: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for doc_number, title, doc_type, field, chunk_id, breadcrumb, raw, tokens in cur:
        found = article_of(breadcrumb)
        if not found:
            continue
        code, heading = found
        item = articles.setdefault(
            (doc_number, code),
            {
                "doc_number": doc_number,
                "doc_title": title,
                "doc_type": doc_type,
                "legal_field": (field or "").split(",")[0].strip(),
                "article_code": code,
                "heading": heading,
                "era": era_of(doc_number),
                "tokens": 0,
                "parts": [],
                "chunk_ids": [],
            },
        )
        item["tokens"] += tokens or 0
        item["parts"].append(body_of(raw, breadcrumb))
        item["chunk_ids"].append(chunk_id)
    return articles


def seed(item: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("doc_number", "doc_title", "doc_type", "legal_field", "article_code", "heading", "era", "tokens")
    return {**{k: item[k] for k in keep}, "text": "\n".join(dict.fromkeys(item["parts"]))}


def is_substantive(item: Dict[str, Any]) -> bool:
    return (
        MIN_ARTICLE_TOKENS <= item["tokens"] <= MAX_ARTICLE_TOKENS
        and not RE_BOILERPLATE.search(item["heading"])
    )


def sample_articles(articles: Dict[Tuple[str, str], Dict[str, Any]], total: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Round-robin over era and legal field inside each document type, one article per document."""
    picked: List[Dict[str, Any]] = []
    for doc_type, share in TYPE_QUOTA.items():
        strata: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        for item in articles.values():
            if item["doc_type"] == doc_type and is_substantive(item):
                strata[(item["era"], item["legal_field"])][item["doc_number"]].append(item)
        keys = sorted(strata)
        rng.shuffle(keys)
        quota, used = round(total * share), set()
        while quota > 0 and keys:
            for key in list(keys):
                docs = [d for d in sorted(strata[key]) if d not in used]
                if not docs:
                    keys.remove(key)
                    continue
                doc = rng.choice(docs)
                used.add(doc)
                picked.append(seed(rng.choice(sorted(strata[key][doc], key=lambda a: a["article_code"]))))
                quota -= 1
                if quota == 0:
                    break
    return picked


def sample_pairs(cur, articles: Dict[Tuple[str, str], Dict[str, Any]], total: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Article pairs where one article cites a specific article of another document."""
    cur.execute(
        """
        SELECT DISTINCT src_d.doc_number, src.breadcrumb, tgt_d.doc_number, tgt.breadcrumb
        FROM provision_references pr
        JOIN legal_chunks src ON src.chunk_id = pr.source_chunk_id
        JOIN legal_chunks tgt ON tgt.chunk_id = pr.resolved_target_chunk_id
        JOIN legal_documents src_d ON src_d.doc_id = src.doc_id
        JOIN legal_documents tgt_d ON tgt_d.doc_id = tgt.doc_id
        WHERE src.doc_id <> tgt.doc_id
        """
    )
    rows = sorted(cur.fetchall())
    rng.shuffle(rows)
    pairs: List[Dict[str, Any]] = []
    seen_docs = set()
    for src_doc, src_crumb, tgt_doc, tgt_crumb in rows:
        src_art, tgt_art = article_of(src_crumb), article_of(tgt_crumb)
        if not src_art or not tgt_art:
            continue
        src, tgt = articles.get((src_doc, src_art[0])), articles.get((tgt_doc, tgt_art[0]))
        if not src or not tgt or not is_substantive(src) or not is_substantive(tgt):
            continue
        if src_doc in seen_docs or tgt_doc in seen_docs:
            continue
        seen_docs.update((src_doc, tgt_doc))
        pairs.append({"source": seed(src), "target": seed(tgt)})
        if len(pairs) == total:
            break
    return pairs


def sample_temporal(cur, total: int, rng: random.Random) -> List[Dict[str, Any]]:
    """The relation source is the newer document. Its title names the document it replaces."""
    cur.execute(
        """
        SELECT old.doc_number, old.doc_title, new.doc_number, new.doc_title
        FROM document_relations r
        JOIN legal_documents new ON new.doc_id = r.source_doc_id
        JOIN legal_documents old ON old.doc_id = r.target_doc_id
        WHERE r.relation_type = 'replaces' AND r.note LIKE 'title:%%'
        ORDER BY old.doc_number, new.doc_number
        """
    )
    rows = [r for r in cur.fetchall() if r[0] != r[2]]
    rng.shuffle(rows)
    return [
        {"old_doc": o, "old_title": ot, "new_doc": n, "new_title": nt, "verb": "bãi bỏ" if "bãi bỏ" in nt.lower() else "thay thế"}
        for o, ot, n, nt in rows[:total]
    ]


def sample_structural(articles: Dict[Tuple[str, str], Dict[str, Any]], total: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Only documents numbered 1..N with no gap, so the count cannot come from a parsing miss."""
    per_doc: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for item in articles.values():
        per_doc[item["doc_number"]].append(item)
    eligible = []
    for doc_number, items in sorted(per_doc.items()):
        codes = [a["article_code"].split()[1] for a in items]
        if not all(c.isdigit() for c in codes) or len(codes) < 5:
            continue
        if RE_AMENDING_TITLE.search(items[0]["doc_title"]):
            continue  # quoted articles of the amended text would inflate the count
        if sorted(int(c) for c in codes) == list(range(1, len(codes) + 1)):
            eligible.append((doc_number, items[0]["doc_title"], items[0]["doc_type"], len(codes)))
    rng.shuffle(eligible)
    return [
        {"doc_number": d, "doc_title": t, "doc_type": ty, "article_count": n, "eligible_docs": len(eligible)}
        for d, t, ty, n in eligible[:total]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/v5_1_phase2/seeds.json")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--articles", type=int, default=320)
    parser.add_argument("--pairs", type=int, default=90)
    parser.add_argument("--temporal", type=int, default=15)
    parser.add_argument("--structural", type=int, default=15)
    args = parser.parse_args()

    load_dotenv(".env")
    conn = psycopg2.connect(os.environ["DATABASE_DSN"])
    conn.set_session(readonly=True)
    rng = random.Random(args.seed)
    with conn.cursor(name="silver_chunks") as stream:
        stream.itersize = 5000
        articles = load_articles(stream)
    with conn.cursor() as cur:
        seeds = {
            "seed": args.seed,
            "articles": sample_articles(articles, args.articles, rng),
            "pairs": sample_pairs(cur, articles, args.pairs, rng),
            "temporal": sample_temporal(cur, args.temporal, rng),
            "structural": sample_structural(articles, args.structural, rng),
        }
    conn.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(seeds, f, ensure_ascii=False, indent=1)

    picked = seeds["articles"]
    print("articles in corpus:", len(articles), "substantive:", sum(map(is_substantive, articles.values())))
    print({k: len(v) for k, v in seeds.items() if isinstance(v, list)})
    print("type:", collections.Counter(a["doc_type"] for a in picked))
    print("era:", collections.Counter(a["era"] for a in picked))
    print("fields:", len({a["legal_field"] for a in picked}), collections.Counter(a["legal_field"] for a in picked).most_common(5))


if __name__ == "__main__":
    main()
