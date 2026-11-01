import argparse
import os
import re
import time
import uuid
from typing import Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

RE_DOC_NUM = re.compile(r"\b(\d+[a-zA-Z]?/\d{4}/[A-ZĐa-zđ0-9\-]+)\b")
RE_OWN_ARTICLE = re.compile(r"\bĐiều\s+(\d+[a-zA-ZđĐ]?)\b", re.IGNORECASE)

RE_CROSS_REF = re.compile(
    r"\bĐiều\s+(\d+[a-zA-ZđĐ]?)\b[^.;:\n]{0,90}?\b(\d+[a-zA-Z]?/\d{4}/[A-ZĐa-zđ0-9\-]+)\b",
    re.IGNORECASE,
)
RE_INTRA_REF = re.compile(
    r"(?:tại|theo|khoản\s+\d+\s+)Điều\s+(\d+[a-zA-ZđĐ]?)(?:\s+(?:của\s+)?(?:Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư|Pháp\s+lệnh|Quyết\s+định)?\s*này)?\b",
    re.IGNORECASE,
)
RE_DOC_REL_CONTEXT = re.compile(
    r"(sửa\s+đổi,\s*bổ\s+sung|sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ|hướng\s+dẫn)[^.;:\n]{0,120}?\b(\d+[a-zA-Z]?/\d{4}/[A-ZĐa-zđ0-9\-]+)\b",
    re.IGNORECASE,
)


def classify_relation(verb: str) -> str:
    v = verb.lower()
    if "sửa đổi" in v:
        return "amends"
    if "bổ sung" in v:
        return "supplements"
    if "thay thế" in v or "bãi bỏ" in v:
        return "replaces"
    if "hướng dẫn" in v:
        return "guides"
    return "cites"


def norm_article(num_part: str) -> str:
    return f"Điều {num_part.lower()}" if num_part[-1:].isalpha() else f"Điều {num_part}"


def populate(dry_run: bool = False, batch_size: int = 5000) -> None:
    t0 = time.perf_counter()
    load_dotenv(".env")
    dsn = os.getenv("DATABASE_DSN")
    if not dsn:
        raise ValueError("DATABASE_DSN is not set in .env")

    conn = psycopg2.connect(dsn)
    doc_num_to_id: Dict[str, str] = {}
    doc_id_to_num: Dict[str, str] = {}
    relations_map: Dict[Tuple[str, str], Tuple[str, str, str, str, Optional[str]]] = {}

    with conn.cursor() as cur:
        cur.execute("SELECT doc_id::text, doc_number, doc_title FROM legal_documents")
        docs = cur.fetchall()

    for doc_id, doc_num, doc_title in docs:
        if not doc_num:
            continue
        clean_num = doc_num.strip().upper()
        doc_num_to_id[clean_num] = doc_id
        doc_id_to_num[doc_id] = clean_num

    for doc_id, doc_num, doc_title in docs:
        src_num = (doc_num or "").strip().upper()
        title = doc_title or ""
        for verb, target_raw in RE_DOC_REL_CONTEXT.findall(title):
            tgt_num = target_raw.upper()
            tgt_id = doc_num_to_id.get(tgt_num)
            if tgt_id and tgt_id != doc_id:
                rel_type = classify_relation(verb)
                rel_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"rel:{doc_id}:{tgt_id}:{rel_type}"))
                relations_map[(doc_id, tgt_id)] = (rel_id, doc_id, tgt_id, rel_type, f"title:{src_num}->{tgt_num}")

    print(f"[1/4] Loaded {len(doc_num_to_id)} documents; found {len(relations_map)} title-level relations.")

    article_to_chunk: Dict[Tuple[str, str], str] = {}
    pending_refs: Set[Tuple[str, str, str]] = set()

    with conn.cursor(name="stream_chunks") as cur:
        cur.itersize = 20000
        cur.execute("SELECT chunk_id::text, doc_id::text, breadcrumb, raw_content FROM legal_chunks")
        scanned = 0
        for cid, doc_id, breadcrumb, raw_content in cur:
            scanned += 1
            own_doc_num = doc_id_to_num.get(doc_id, "")
            if not own_doc_num:
                continue

            m_own = RE_OWN_ARTICLE.search(breadcrumb or "")
            own_art = norm_article(m_own.group(1)) if m_own else ""
            is_whole_article = " > Khoản " not in (breadcrumb or "")
            if own_art and (is_whole_article or (own_doc_num, own_art) not in article_to_chunk):
                article_to_chunk[(own_doc_num, own_art)] = cid

            text = raw_content or ""
            if own_art in ("Điều 1", "Điều 2", "Điều 3") or len(text) < 1200:
                for verb, target_raw in RE_DOC_REL_CONTEXT.findall(text):
                    tgt_num = target_raw.upper()
                    tgt_id = doc_num_to_id.get(tgt_num)
                    if tgt_id and tgt_id != doc_id and (doc_id, tgt_id) not in relations_map:
                        rel_type = classify_relation(verb)
                        rel_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"rel:{doc_id}:{tgt_id}:{rel_type}"))
                        relations_map[(doc_id, tgt_id)] = (
                            rel_id,
                            doc_id,
                            tgt_id,
                            rel_type,
                            f"body:{own_doc_num}->{tgt_num}",
                        )

            for art_num, target_doc_raw in RE_CROSS_REF.findall(text):
                tgt_doc = target_doc_raw.upper()
                tgt_art = norm_article(art_num)
                if tgt_doc == own_doc_num and tgt_art == own_art:
                    continue
                pending_refs.add((cid, tgt_doc, tgt_art))
                tgt_id = doc_num_to_id.get(tgt_doc)
                if tgt_id and tgt_id != doc_id and (doc_id, tgt_id) not in relations_map:
                    rel_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"rel:{doc_id}:{tgt_id}:cites"))
                    relations_map[(doc_id, tgt_id)] = (rel_id, doc_id, tgt_id, "cites", f"ref:{own_doc_num}->{tgt_doc}")

            for art_num in RE_INTRA_REF.findall(text):
                tgt_art = norm_article(art_num)
                if tgt_art != own_art:
                    pending_refs.add((cid, own_doc_num, tgt_art))

    ref_rows: List[Tuple[str, str, str, str, Optional[str]]] = []
    resolved_count = 0
    for src_cid, tgt_doc, tgt_art in pending_refs:
        resolved_cid = article_to_chunk.get((tgt_doc, tgt_art))
        if resolved_cid == src_cid:
            continue
        if resolved_cid:
            resolved_count += 1
        ref_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"ref:{src_cid}:{tgt_doc}:{tgt_art}"))
        ref_rows.append((ref_id, src_cid, tgt_doc, tgt_art, resolved_cid))

    rel_rows = list(relations_map.values())
    print(f"[2/4] Scanned {scanned} chunks in {time.perf_counter() - t0:.1f}s.")
    print(f"      Extracted {len(rel_rows)} document_relations and {len(ref_rows)} provision_references ({resolved_count} resolved to exact chunk UUIDs).")

    if dry_run:
        print("[DRY RUN] Skipping database writes.")
        conn.close()
        return

    with conn.cursor() as cur:
        for i in range(0, len(rel_rows), batch_size):
            batch = rel_rows[i : i + batch_size]
            execute_values(
                cur,
                """
                INSERT INTO document_relations (relation_id, source_doc_id, target_doc_id, relation_type, note)
                VALUES %s
                ON CONFLICT (relation_id) DO NOTHING
                """,
                batch,
                page_size=batch_size,
            )
        conn.commit()
    print(f"[3/4] Inserted {len(rel_rows)} rows into document_relations.")

    with conn.cursor() as cur:
        for i in range(0, len(ref_rows), batch_size):
            batch = ref_rows[i : i + batch_size]
            execute_values(
                cur,
                """
                INSERT INTO provision_references (ref_id, source_chunk_id, target_doc_number, target_article_code, resolved_target_chunk_id)
                VALUES %s
                ON CONFLICT (ref_id) DO NOTHING
                """,
                batch,
                page_size=batch_size,
            )
            conn.commit()
    print(f"[4/4] Inserted {len(ref_rows)} rows into provision_references in {time.perf_counter() - t0:.1f}s total.")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Extract and count edges without writing to PostgreSQL")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    populate(dry_run=args.dry_run, batch_size=args.batch_size)

