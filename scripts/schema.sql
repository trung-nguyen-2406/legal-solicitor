CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS legal_documents (
    doc_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_number VARCHAR(100) UNIQUE NOT NULL,
    doc_title TEXT NOT NULL,
    doc_type VARCHAR(50) NOT NULL,
    issuing_body VARCHAR(100),
    signer_name VARCHAR(100),
    issued_date DATE,
    effective_date DATE,
    expiry_date DATE,
    effective_status VARCHAR(30) DEFAULT 'active',
    legal_field VARCHAR(100),
    content_sha256 CHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS document_sections (
    section_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id UUID REFERENCES legal_documents(doc_id) ON DELETE CASCADE,
    parent_section_id UUID REFERENCES document_sections(section_id) ON DELETE CASCADE,
    level_type VARCHAR(20) NOT NULL,
    section_code VARCHAR(50) NOT NULL,
    title_or_lead_in TEXT,
    full_text TEXT NOT NULL,
    display_order INT NOT NULL,
    status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS document_relations (
    relation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_doc_id UUID REFERENCES legal_documents(doc_id) ON DELETE CASCADE,
    target_doc_id UUID REFERENCES legal_documents(doc_id) ON DELETE CASCADE,
    relation_type VARCHAR(30) NOT NULL,
    target_section_id UUID REFERENCES document_sections(section_id) ON DELETE SET NULL,
    effective_from DATE,
    note TEXT
);

CREATE TABLE IF NOT EXISTS legal_chunks (
    chunk_id UUID PRIMARY KEY,
    doc_id UUID REFERENCES legal_documents(doc_id) ON DELETE CASCADE,
    section_id UUID REFERENCES document_sections(section_id) ON DELETE SET NULL,
    breadcrumb TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    token_count INT NOT NULL,
    chunk_sha256 CHAR(64) NOT NULL,
    is_indexed_qdrant BOOLEAN DEFAULT FALSE,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS provision_references (
    ref_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_chunk_id UUID REFERENCES legal_chunks(chunk_id) ON DELETE CASCADE,
    target_doc_number VARCHAR(100) NOT NULL,
    target_article_code VARCHAR(50) NOT NULL,
    resolved_target_chunk_id UUID REFERENCES legal_chunks(chunk_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS crawl_jobs (
    job_id SERIAL PRIMARY KEY,
    trigger_source VARCHAR(30) NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'RUNNING'
);

CREATE TABLE IF NOT EXISTS crawl_job_items (
    item_id BIGSERIAL PRIMARY KEY,
    job_id INT REFERENCES crawl_jobs(job_id) ON DELETE CASCADE,
    doc_id UUID REFERENCES legal_documents(doc_id) ON DELETE CASCADE,
    action_taken VARCHAR(30) NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_ip VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_logs (
    request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    raw_query TEXT NOT NULL,
    normalized_query TEXT,
    extracted_doc_number VARCHAR(100),
    extracted_article VARCHAR(50),
    cache_hit BOOLEAN DEFAULT FALSE,
    abstained BOOLEAN DEFAULT FALSE,
    generated_answer TEXT,
    llm_model_name VARCHAR(50),
    retrieval_ms INT,
    rerank_ms INT,
    generation_ms INT,
    user_rating SMALLINT,
    user_feedback_comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_retrieved_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID REFERENCES query_logs(request_id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES legal_chunks(chunk_id) ON DELETE CASCADE,
    rank_order INT NOT NULL,
    dense_score FLOAT,
    sparse_score FLOAT,
    rrf_score FLOAT,
    rerank_score FLOAT,
    passed_threshold BOOLEAN DEFAULT FALSE
);
