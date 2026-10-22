import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

class LegalChunkRecord(BaseModel):
    chunk_id: str = Field(..., description="UUID5 generated from doc_number + article_id + clause_id")
    doc_id: str
    doc_number: str
    doc_title: str
    doc_type: str
    issuing_body: Optional[str] = None
    issued_year: int
    effective_status: str = "active"
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    article_id: str
    clause_id: str = "ALL"
    breadcrumb: str
    raw_content: str
    token_count: int = 0
    chunk_sha256: str
    referenced_provisions: List[str] = []

    @classmethod
    def generate_chunk_id(cls, doc_number: str, article_id: str, clause_id: str) -> str:
        key = f"{doc_number}_{article_id}_{clause_id}".strip()
        return str(uuid.uuid5(uuid.NAMESPACE_OID, key))
