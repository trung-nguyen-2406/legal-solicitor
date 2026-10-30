import os
import re
import uuid
from typing import Any, Dict, List, Optional, Set
from dotenv import load_dotenv
import psycopg2

RE_ARTICLE_CODE = re.compile(r"\b(Điều\s+\d+[a-zA-ZđĐ]?)\b", re.IGNORECASE)
RE_DOC_YEAR = re.compile(r"/(\d{4})/")
AMENDING_RELATIONS = (
    "amends",
    "supplements",
    "amends_and_supplements",
    "replaces",
    "sửa đổi",
    "bổ sung",
    "sửa đổi, bổ sung",
    "thay thế",
)


def _valid_uuids(items: List[str]) -> List[str]:
    valid: List[str] = []
    for item in items:
        if not item:
            continue
        try:
            valid.append(str(uuid.UUID(str(item))))
        except ValueError:
            continue
    return valid


def _extract_article_code(breadcrumb: str, raw_content: str) -> str:
    m = RE_ARTICLE_CODE.search(breadcrumb or "") or RE_ARTICLE_CODE.search(raw_content or "")
    if not m:
        return ""
    parts = m.group(1).split()
    return f"Điều {parts[1]}" if len(parts) >= 2 else m.group(1)


def _extract_year(pg_year: Optional[int], doc_number: str, fallback: int = 2015) -> int:
    if pg_year:
        return int(pg_year)
    m = RE_DOC_YEAR.search(doc_number or "")
    return int(m.group(1)) if m else fallback


def _row_to_chunk(row: Any) -> Dict[str, Any]:
    cid, raw_content, breadcrumb, doc_id, doc_number, doc_title, doc_type, pg_year, eff_status, legal_field = row
    art_code = _extract_article_code(breadcrumb, raw_content)
    yr = _extract_year(pg_year, doc_number)
    clause_code = "ALL"
    if breadcrumb and " > Khoản " in breadcrumb:
        clause_code = "Khoản " + breadcrumb.split(" > Khoản ")[-1].strip()
    return {
        "chunk_id": cid,
        "doc_id": doc_id,
        "doc_number": doc_number or "",
        "doc_title": doc_title or "",
        "doc_type": doc_type or "",
        "issued_year": yr,
        "effective_status": eff_status or "active",
        "legal_field": legal_field or "",
        "article_code": art_code,
        "clause_code": clause_code,
        "breadcrumb": breadcrumb or "",
        "raw_content": raw_content or breadcrumb or "",
        "anchor_tag": f"[{doc_number or ''}:{art_code}]",
    }


class DatabaseService:
    def __init__(self, dsn: Optional[str] = None):
        if not dsn:
            load_dotenv(".env")
            dsn = os.getenv("DATABASE_DSN")
        self.dsn = dsn
        self._conn = None

    def _get_conn(self):
        if not self.dsn:
            return None
        if self._conn is None or self._conn.closed != 0:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
        return self._conn

    def hydrate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        conn = self._get_conn()
        valid_ids = _valid_uuids([c.get("chunk_id", "") for c in chunks])
        pg_map: Dict[str, Dict[str, Any]] = {}

        if conn is not None and valid_ids:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT lc.chunk_id::text, lc.raw_content, lc.breadcrumb,
                           ld.doc_id::text, ld.doc_number, ld.doc_title, ld.doc_type,
                           EXTRACT(YEAR FROM ld.issued_date)::int,
                           ld.effective_status, ld.legal_field
                    FROM legal_chunks lc
                    JOIN legal_documents ld ON lc.doc_id = ld.doc_id
                    WHERE lc.chunk_id = ANY(%s::uuid[])
                    """,
                    (valid_ids,),
                )
                for row in cur.fetchall():
                    pg_map[row[0]] = _row_to_chunk(row)

        hydrated: List[Dict[str, Any]] = []
        for c in chunks:
            item = dict(c)
            cid = str(item.get("chunk_id", ""))
            if cid in pg_map:
                pg_item = pg_map[cid]
                item["raw_content"] = pg_item["raw_content"]
                item["doc_title"] = item.get("doc_title") or pg_item["doc_title"]
                item["legal_field"] = item.get("legal_field") or pg_item["legal_field"]
                item["doc_id"] = item.get("doc_id") or pg_item["doc_id"]
                item["doc_number"] = item.get("doc_number") or pg_item["doc_number"]
                item["article_code"] = item.get("article_code") or pg_item["article_code"]
                item["issued_year"] = item.get("issued_year") or pg_item["issued_year"]
            else:
                item["raw_content"] = item.get("raw_content") or item.get("breadcrumb") or ""
                if not item.get("article_code"):
                    item["article_code"] = _extract_article_code(item.get("breadcrumb", ""), item["raw_content"])
                if not item.get("issued_year"):
                    item["issued_year"] = _extract_year(None, item.get("doc_number", ""))

            if not item.get("anchor_tag"):
                item["anchor_tag"] = f"[{item.get('doc_number', '')}:{item.get('article_code', '')}]"
            hydrated.append(item)

        return hydrated

    def fetch_amending_provisions(
        self,
        doc_ids: List[str],
        existing_chunk_ids: Set[str],
        article_codes: Optional[List[str]] = None,
        limit: int = 2,
    ) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        valid_docs = _valid_uuids(doc_ids)
        if conn is None or not valid_docs:
            return []

        art_patterns = [f"%{a}%" for a in (article_codes or []) if a]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (lc.chunk_id)
                       lc.chunk_id::text, lc.raw_content, lc.breadcrumb,
                       ld.doc_id::text, ld.doc_number, ld.doc_title, ld.doc_type,
                       EXTRACT(YEAR FROM ld.issued_date)::int,
                       ld.effective_status, ld.legal_field
                FROM document_relations dr
                JOIN legal_documents ld
                  ON ld.doc_id = CASE
                       WHEN dr.target_doc_id = ANY(%s::uuid[]) THEN dr.source_doc_id
                       ELSE dr.target_doc_id
                     END
                JOIN legal_chunks lc ON lc.doc_id = ld.doc_id
                WHERE (dr.target_doc_id = ANY(%s::uuid[]) OR dr.source_doc_id = ANY(%s::uuid[]))
                  AND LOWER(dr.relation_type) = ANY(%s)
                  AND ld.effective_status != 'expired'
                  AND (
                      (dr.target_section_id IS NOT NULL AND lc.section_id = dr.target_section_id)
                      OR (%s::text[] != '{}'::text[] AND (lc.breadcrumb ILIKE ANY(%s) OR lc.raw_content ILIKE ANY(%s)))
                      OR dr.target_section_id IS NULL
                  )
                LIMIT %s
                """,
                (
                    valid_docs,
                    valid_docs,
                    valid_docs,
                    list(AMENDING_RELATIONS),
                    art_patterns,
                    art_patterns,
                    art_patterns,
                    limit + len(existing_chunk_ids),
                ),
            )
            rows = cur.fetchall()

        pinned: List[Dict[str, Any]] = []
        for row in rows:
            if row[0] in existing_chunk_ids:
                continue
            item = _row_to_chunk(row)
            item["is_pinned_amendment"] = True
            pinned.append(item)
            if len(pinned) >= limit:
                break
        return pinned

    def expand_bidirectional(self, chunks: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        if conn is None or not chunks:
            return []

        existing_ids = {str(c.get("chunk_id", "")) for c in chunks}
        valid_cids = _valid_uuids(list(existing_ids))
        valid_docs = _valid_uuids([str(c.get("doc_id", "")) for c in chunks])
        doc_nums = [c["doc_number"] for c in chunks if c.get("doc_number")]
        art_codes = [c["article_code"] for c in chunks if c.get("article_code")]

        expanded: List[Dict[str, Any]] = []
        with conn.cursor() as cur:
            if valid_cids or (doc_nums and art_codes):
                cur.execute(
                    """
                    SELECT DISTINCT lc.chunk_id::text, lc.raw_content, lc.breadcrumb,
                           ld.doc_id::text, ld.doc_number, ld.doc_title, ld.doc_type,
                           EXTRACT(YEAR FROM ld.issued_date)::int,
                           ld.effective_status, ld.legal_field
                    FROM provision_references pr
                    JOIN legal_chunks lc
                      ON lc.chunk_id = pr.resolved_target_chunk_id
                      OR lc.chunk_id = pr.source_chunk_id
                    JOIN legal_documents ld ON lc.doc_id = ld.doc_id
                    WHERE ld.effective_status != 'expired'
                      AND (
                          pr.source_chunk_id = ANY(%s::uuid[])
                          OR pr.resolved_target_chunk_id = ANY(%s::uuid[])
                          OR (pr.target_doc_number = ANY(%s) AND pr.target_article_code = ANY(%s))
                      )
                    LIMIT %s
                    """,
                    (valid_cids, valid_cids, doc_nums, art_codes, limit + len(existing_ids)),
                )
                for row in cur.fetchall():
                    if row[0] not in existing_ids:
                        existing_ids.add(row[0])
                        expanded.append(_row_to_chunk(row))
                        if len(expanded) >= limit:
                            return expanded

            if len(expanded) < limit and valid_docs:
                rem = limit - len(expanded)
                cur.execute(
                    """
                    SELECT DISTINCT lc.chunk_id::text, lc.raw_content, lc.breadcrumb,
                           ld.doc_id::text, ld.doc_number, ld.doc_title, ld.doc_type,
                           EXTRACT(YEAR FROM ld.issued_date)::int,
                           ld.effective_status, ld.legal_field
                    FROM document_relations dr
                    JOIN legal_documents ld
                      ON ld.doc_id = CASE
                           WHEN dr.target_doc_id = ANY(%s::uuid[]) THEN dr.source_doc_id
                           ELSE dr.target_doc_id
                         END
                    JOIN legal_chunks lc
                      ON lc.doc_id = ld.doc_id
                     AND (dr.target_section_id IS NULL OR lc.section_id = dr.target_section_id)
                    WHERE (dr.source_doc_id = ANY(%s::uuid[]) OR dr.target_doc_id = ANY(%s::uuid[]))
                      AND ld.effective_status != 'expired'
                    LIMIT %s
                    """,
                    (valid_docs, valid_docs, valid_docs, rem + len(existing_ids)),
                )
                for row in cur.fetchall():
                    if row[0] not in existing_ids:
                        existing_ids.add(row[0])
                        expanded.append(_row_to_chunk(row))
                        if len(expanded) >= limit:
                            break

        return expanded

    def fetch_amendment_lineage_cte(self, doc_number: str, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        if conn is None or not doc_number:
            return []

        with conn.cursor() as cur:
            cur.execute(
                """
                WITH RECURSIVE lineage AS (
                    SELECT ld.doc_id, 0 AS depth, ARRAY[ld.doc_id] AS visited
                    FROM legal_documents ld
                    WHERE ld.doc_number = %s
                    UNION ALL
                    SELECT nxt.doc_id, l.depth + 1, l.visited || nxt.doc_id
                    FROM lineage l
                    JOIN document_relations dr
                      ON dr.source_doc_id = l.doc_id OR dr.target_doc_id = l.doc_id
                    JOIN legal_documents nxt
                      ON nxt.doc_id = CASE
                           WHEN dr.source_doc_id = l.doc_id THEN dr.target_doc_id
                           ELSE dr.source_doc_id
                         END
                    WHERE l.depth < 5
                      AND NOT (nxt.doc_id = ANY(l.visited))
                )
                SELECT DISTINCT lc.chunk_id::text, lc.raw_content, lc.breadcrumb,
                       ld.doc_id::text, ld.doc_number, ld.doc_title, ld.doc_type,
                       EXTRACT(YEAR FROM ld.issued_date)::int,
                       ld.effective_status, ld.legal_field
                FROM lineage l
                JOIN legal_documents ld ON ld.doc_id = l.doc_id
                JOIN legal_chunks lc ON lc.doc_id = ld.doc_id
                WHERE l.depth > 0
                  AND ld.effective_status != 'expired'
                LIMIT %s
                """,
                (doc_number, limit),
            )
            rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            item = _row_to_chunk(row)
            item["rerank_score"] = 0.85
            results.append(item)
        return results

