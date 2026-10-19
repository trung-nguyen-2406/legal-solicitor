import re
import hashlib
import unicodedata
from typing import List, Dict, Any, Optional
from apps.schemas.chunk_models import LegalChunkRecord

RE_CHAPTER = re.compile(r"^Chương\s+([IVXLCDM\d]+)[\.\:\s]*(.*)$", re.IGNORECASE)
RE_SECTION = re.compile(r"^Mục\s+(\d+)[\.\:\s]*(.*)$", re.IGNORECASE)
RE_ARTICLE = re.compile(r"^Điều\s+(\d+[a-zđ]?)\.\s*(.*)$", re.IGNORECASE)
RE_CLAUSE = re.compile(r"^(\d+)\.\s+(.*)$")
RE_PROVISION_REF = re.compile(r"\bĐiều\s+\d+[a-zđ]?\b", re.IGNORECASE)

class LegalParserService:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return int(len(text.split())* 1.3)

    def _extract_references(self, text: str) -> List[str]:
        return sorted(list(set(RE_PROVISION_REF.findall(text))))

    def _flush_article(
        self,
        doc_meta: Dict[str, Any],
        chapter: Optional[str],
        section: Optional[str],
        article_id: str,
        article_title: str,
        lead_in: str,
        clauses: List[Dict[str, str]],
    ) -> List[LegalChunkRecord]:
        chunks = []
        doc_number = doc_meta["doc_number"]
        breadcrumb_parts = [doc_number]
        if chapter:
            breadcrumb_parts.append(chapter)
        if section:
            breadcrumb_parts.append(section)
        breadcrumb_parts.append(f"{article_id}. {article_title}".strip())
        base_breadcrumb = " > ".join(breadcrumb_parts)

        # Assemble 
        full_lines = []
        if article_title:
            full_lines.append(f"{article_id}. {article_title}")
        else:
            full_lines.append(article_id)
        if lead_in:
            full_lines.append(lead_in)
        for cl in clauses:
            full_lines.append(f"{cl['code']}. {cl['text']}")
        full_article_content = "\n".join(full_lines).strip()
        total_tokens = self._count_tokens(full_article_content)

        # Case 1 Article < 1024 token limit 
        if total_tokens <= 1024 or not clauses:
            chunk_id = LegalChunkRecord.generate_chunk_id(doc_number, article_id, "ALL")
            chunks.append(
                LegalChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_meta["doc_id"],
                    doc_number=doc_number,
                    doc_title=doc_meta["doc_title"],
                    doc_type=doc_meta["doc_type"],
                    issuing_body=doc_meta.get("issuing_body"),
                    issued_year=doc_meta["issued_year"],
                    effective_status=doc_meta.get("effective_status", "active"),
                    chapter_id=chapter,
                    section_id=section,
                    article_id=article_id,
                    clause_id="ALL",
                    breadcrumb=base_breadcrumb,
                    raw_content=full_article_content,
                    token_count=total_tokens,
                    chunk_sha256=hashlib.sha256(full_article_content.encode("utf-8")).hexdigest(),
                    referenced_provisions=self._extract_references(full_article_content),
                )
            )
            return chunks

        # Case 2 Article > 1024 tokens
        for cl in clauses:
            clause_id = f"Khoản {cl['code']}"
            clause_content = (
                f"{base_breadcrumb}\n{article_id}. {article_title}\n{lead_in}\n{cl['code']}. {cl['text']}".strip()
            )
            c_tokens = self._count_tokens(clause_content)
            chunk_id = LegalChunkRecord.generate_chunk_id(doc_number, article_id, clause_id)
            chunks.append(
                LegalChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_meta["doc_id"],
                    doc_number=doc_number,
                    doc_title=doc_meta["doc_title"],
                    doc_type=doc_meta["doc_type"],
                    issuing_body=doc_meta.get("issuing_body"),
                    issued_year=doc_meta["issued_year"],
                    effective_status=doc_meta.get("effective_status", "active"),
                    chapter_id=chapter,
                    section_id=section,
                    article_id=article_id,
                    clause_id=clause_id,
                    breadcrumb=f"{base_breadcrumb} > {clause_id}",
                    raw_content=clause_content,
                    token_count=c_tokens,
                    chunk_sha256=hashlib.sha256(clause_content.encode("utf-8")).hexdigest(),
                    referenced_provisions=self._extract_references(clause_content),
                )
            )
        return chunks

    def parse_document(self, text: str, doc_meta: Dict[str, Any]) -> List[LegalChunkRecord]:
        text = unicodedata.normalize("NFC", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        current_chapter: Optional[str] = None
        current_section: Optional[str] = None
        current_article_id: Optional[str] = None
        current_article_title: str = ""
        article_lead_in: List[str] = []
        clauses: List[Dict[str, str]] = []
        current_clause_idx: Optional[int] = None
        records: List[LegalChunkRecord] = []

        for line in lines:
            m_chap = RE_CHAPTER.match(line)
            if m_chap:
                current_chapter = f"Chương {m_chap.group(1)} {m_chap.group(2)}".strip()
                current_section = None
                continue

            m_sec = RE_SECTION.match(line)
            if m_sec:
                current_section = f"Mục {m_sec.group(1)} {m_sec.group(2)}".strip()
                continue

            m_art = RE_ARTICLE.match(line)
            if m_art:
                if current_article_id:
                    records.extend(
                        self._flush_article(
                            doc_meta,
                            current_chapter,
                            current_section,
                            current_article_id,
                            current_article_title,
                            "\n".join(article_lead_in),
                            clauses,
                        )
                    )
                current_article_id = f"Điều {m_art.group(1)}"
                current_article_title = m_art.group(2).strip()
                article_lead_in = []
                clauses = []
                current_clause_idx = None
                continue

            if current_article_id:
                m_cl = RE_CLAUSE.match(line)
                if m_cl:
                    clauses.append({"code": m_cl.group(1), "text": m_cl.group(2)})
                    current_clause_idx = len(clauses) - 1
                elif current_clause_idx is not None:
                    clauses[current_clause_idx]["text"] += f"\n{line}"
                else:
                    article_lead_in.append(line)

        if current_article_id:
            records.extend(
                self._flush_article(
                    doc_meta,
                    current_chapter,
                    current_section,
                    current_article_id,
                    current_article_title,
                    "\n".join(article_lead_in),
                    clauses,
                )
            )

        return records
