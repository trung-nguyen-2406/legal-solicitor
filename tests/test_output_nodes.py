import asyncio
import hashlib
import os
import sys
import time
from typing import Any, Dict
import uuid
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
import psycopg2

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
load_dotenv(".env")

from apps.services.cache_service import (
    NEGATIVE_CACHE_TAG_KEY,
    cache_key_for_sha,
    doc_tag_key,
    invalidate_by_doc,
)
from apps.services.llm_service import (
    INVENTORY_TABLE_MAX_TOKENS,
    MAX_CONTEXT_TOKENS,
    build_irac_context_and_citations,
    estimate_tokens,
)
from apps.services.resources import get_db_service, get_redis_client
from apps.workflow.nodes_output import (
    AbstainWithReasonNode,
    ConfidenceGate,
    GenerateHierarchicalInventoryTableNode,
    IRACGenerateNode,
    ReturnResponseNode,
    SetCacheWithDocTagsNode,
    SetGlobalNegativeCacheNode,
    WriteAuditLogToPostgresNode,
    flush_pending_audit_tasks,
    route_after_confidence,
)
from apps.workflow.state import LegalRAGState

_SHARED: Dict[str, Any] = {}


def _build_full_output_graph():
    builder = StateGraph(LegalRAGState)
    builder.add_node("ConfidenceGate", ConfidenceGate)
    builder.add_node("GenerateHierarchicalInventoryTableNode", GenerateHierarchicalInventoryTableNode)
    builder.add_node("IRACGenerateNode", IRACGenerateNode)
    builder.add_node("SetCacheWithDocTagsNode", SetCacheWithDocTagsNode)
    builder.add_node("AbstainWithReasonNode", AbstainWithReasonNode)
    builder.add_node("SetGlobalNegativeCacheNode", SetGlobalNegativeCacheNode)
    builder.add_node("WriteAuditLogToPostgresNode", WriteAuditLogToPostgresNode)
    builder.add_node("ReturnResponseNode", ReturnResponseNode)

    builder.add_edge(START, "ConfidenceGate")
    builder.add_conditional_edges(
        "ConfidenceGate",
        route_after_confidence,
        {
            "grounded_inventory": "GenerateHierarchicalInventoryTableNode",
            "grounded_direct_irac": "IRACGenerateNode",
            "ungrounded_abstain": "AbstainWithReasonNode",
        },
    )
    builder.add_edge("GenerateHierarchicalInventoryTableNode", "IRACGenerateNode")
    builder.add_edge("IRACGenerateNode", "SetCacheWithDocTagsNode")
    builder.add_edge("SetCacheWithDocTagsNode", "WriteAuditLogToPostgresNode")
    builder.add_edge("AbstainWithReasonNode", "SetGlobalNegativeCacheNode")
    builder.add_edge("SetGlobalNegativeCacheNode", "WriteAuditLogToPostgresNode")
    builder.add_edge("WriteAuditLogToPostgresNode", "ReturnResponseNode")
    builder.add_edge("ReturnResponseNode", END)
    return builder.compile()


def _fetch_real_chunk() -> Dict[str, Any]:
    if "real_chunk" in _SHARED:
        return dict(_SHARED["real_chunk"])
    conn = get_db_service()._get_conn()
    assert conn is not None, "DATABASE_DSN connection unavailable!"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT lc.chunk_id::text, lc.raw_content, lc.breadcrumb,
                   ld.doc_number, ld.doc_title, EXTRACT(YEAR FROM ld.issued_date)::int
            FROM legal_chunks lc
            JOIN legal_documents ld ON lc.doc_id = ld.doc_id
            WHERE ld.doc_number = '50/2014/QH13' AND lc.breadcrumb ILIKE '%Điều 21%'
            LIMIT 1
            """
        )
        row = cur.fetchone()
    assert row is not None, "Real chunk for 50/2014/QH13:Điều 21 not found in Neon PG!"
    chunk = {
        "chunk_id": row[0],
        "raw_content": row[1],
        "breadcrumb": row[2],
        "doc_number": row[3],
        "doc_title": row[4],
        "issued_year": row[5] or 2014,
        "article_code": "Điều 21",
        "clause_code": "ALL",
        "rrf_score": 0.033,
        "rerank_score": 0.98,
        "rank_order": 1,
    }
    _SHARED["real_chunk"] = chunk
    return dict(chunk)


def test_confidence_gate_routing_matrix():
    print("=================================================================")
    print("TEST 1: CONFIDENCE GATE & CRITICAL ROUTER MATRIX")
    print("=================================================================")

    big_bank = {f"c{i}": {"chunk_id": f"c{i}"} for i in range(30)}
    small_bank = {f"c{i}": {"chunk_id": f"c{i}"} for i in range(8)}

    scenarios = [
        (
            "Non-aggregation with 30 chunks (MUST route grounded_direct_irac)",
            {"is_aggregation": False, "max_score": 0.85, "top_chunks": [{"chunk_id": "c1"}], "chunk_bank": big_bank},
            "grounded_direct_irac",
        ),
        (
            "Aggregation with >15 chunks",
            {"is_aggregation": True, "max_score": 0.60, "top_chunks": [{"chunk_id": "c1"}], "chunk_bank": big_bank},
            "grounded_inventory",
        ),
        (
            "Aggregation with <=15 chunks",
            {"is_aggregation": True, "max_score": 0.60, "top_chunks": [{"chunk_id": "c1"}], "chunk_bank": small_bank},
            "grounded_direct_irac",
        ),
        (
            "Low score (<0.20) on normal query",
            {"is_aggregation": False, "max_score": 0.17, "top_chunks": [{"chunk_id": "c1"}], "chunk_bank": big_bank},
            "ungrounded_abstain",
        ),
        (
            "Empty top chunks on aggregation query",
            {"is_aggregation": True, "max_score": 0.90, "top_chunks": [], "chunk_bank": big_bank},
            "ungrounded_abstain",
        ),
        (
            "Explicitly abstained upstream",
            {"abstained": True, "max_score": 0.88, "top_chunks": [{"chunk_id": "c1"}], "chunk_bank": big_bank},
            "ungrounded_abstain",
        ),
    ]

    for label, input_state, expected in scenarios:
        gate_out = ConfidenceGate(input_state)
        merged = dict(input_state)
        merged.update(gate_out)
        decision = route_after_confidence(merged)
        assert decision == expected, f"{label}: got {decision}, expected {expected}"
        print(f"[PASSED] {label} -> {decision} (abstained={gate_out.get('abstained')})")
    print()


def test_inventory_table_node_and_context_assembly():
    print("=================================================================")
    print("TEST 2: GenerateHierarchicalInventoryTableNode & CONTEXT BUDGET")
    print("=================================================================")

    synth_bank = {}
    for d in range(6):
        doc_no = f"{20 + d}/2024/NĐ-CP"
        title = f"Nghị định quy định chi tiết một số điều của Luật Đất đai và Xây dựng số {d}"
        for a in range(5):
            cid = f"chk_{d}_{a}"
            art_code = f"Điều {a + 1}"
            synth_bank[cid] = {
                "chunk_id": cid,
                "doc_number": doc_no,
                "doc_title": title,
                "issued_year": 2024,
                "article_code": art_code,
                "raw_content": (
                    f"Căn cứ quy định tại {art_code} văn bản {doc_no}: "
                    "Tổ chức, cá nhân thực hiện hoạt động đầu tư xây dựng phải tuân thủ quy hoạch được duyệt. "
                ) * 10,
                "rerank_score": round(0.92 - d * 0.04 - a * 0.005, 4),
            }

    node_out = GenerateHierarchicalInventoryTableNode({
        "chunk_bank": synth_bank,
        "top_chunks": list(synth_bank.values())[:9],
    })
    table_md = node_out["inventory_table"]
    top_5 = node_out["top_chunks"]
    citations = node_out["citations"]

    tbl_tokens = estimate_tokens(table_md)
    assert tbl_tokens <= INVENTORY_TABLE_MAX_TOKENS, f"Inventory exceeded 600 tokens: {tbl_tokens}"
    assert len(top_5) == 5, f"Expected Top-5 verbatim chunks, got {len(top_5)}"
    assert len(citations) == 30, f"Expected 30 citations, got {len(citations)}"

    context_str, final_citations = build_irac_context_and_citations(
        top_chunks=top_5,
        inventory_table=table_md,
        existing_citations=citations,
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )
    ctx_tokens = estimate_tokens(context_str)
    assert ctx_tokens <= MAX_CONTEXT_TOKENS, f"Context exceeded 2500 tokens: {ctx_tokens}"
    assert len(final_citations) == 30

    print(f"[PASSED] GenerateHierarchicalInventoryTableNode: table_tokens={tbl_tokens}/600, top_chunks={len(top_5)}, citations={len(citations)}")
    print(f"[PASSED] IRAC Context Budget: context_tokens={ctx_tokens}/2500\n{table_md}\n")


def test_live_irac_sse_streaming():
    print("=================================================================")
    print("TEST 3: LIVE LANGGRAPH IRAC SSE STREAMING (OLLAMA QWEN2.5-7B)")
    print("=================================================================")

    async def _run():
        real_chunk = _fetch_real_chunk()
        # Use isolated doc tag on cached copy so later cleanup doesn't touch production doc tags
        builder = StateGraph(LegalRAGState)
        builder.add_node("ConfidenceGate", ConfidenceGate)
        builder.add_node("IRACGenerateNode", IRACGenerateNode)
        builder.add_node("ReturnResponseNode", ReturnResponseNode)
        builder.add_edge(START, "ConfidenceGate")
        builder.add_edge("ConfidenceGate", "IRACGenerateNode")
        builder.add_edge("IRACGenerateNode", "ReturnResponseNode")
        builder.add_edge("ReturnResponseNode", END)
        graph = builder.compile()

        req_id = str(uuid.uuid4())
        t0 = time.time()
        events = []
        token_parts = []

        async for ev in graph.astream(
            {
                "request_id": req_id,
                "query": "Quy định về lưu trữ hồ sơ quy hoạch xây dựng theo Điều 21 Luật Xây dựng?",
                "top_chunks": [real_chunk],
                "chunk_bank": {real_chunk["chunk_id"]: real_chunk},
                "is_aggregation": False,
                "max_score": 0.98,
                "start_time": t0,
                "latency_ms": {"DirectPointGetNode": 12.0},
            },
            stream_mode="custom",
        ):
            events.append(ev)
            if ev.get("event") == "token":
                token_parts.append(ev["data"]["delta"])

        elapsed = time.time() - t0
        full_text = "".join(token_parts)

        assert events[0]["event"] == "metadata"
        assert events[0]["data"]["abstained"] is False
        assert len(events[0]["data"]["citations"]) == 1
        assert len(token_parts) > 10
        assert "[Nguồn 1:" in full_text, f"Missing exact '[Nguồn 1:' citation in answer:\n{full_text}"
        assert events[-1]["event"] == "done"

        print(f"[PASSED] SSE Streaming ({elapsed:.2f}s): events={len(events)}, tokens={len(token_parts)}")
        print(f"       Metadata: {events[0]}")
        print(f"       Metrics : {events[-1]['data']['metrics']}")
        print(f"\n--- FULL STREAMED IRAC ANSWER ---\n{full_text}\n---------------------------------\n")
        _SHARED["answer"] = full_text

    asyncio.run(_run())


def test_redis_positive_tagged_caching_and_invalidation():
    print("=================================================================")
    print("TEST 4: REDIS POSITIVE CACHING WITH ISOLATED DOC-TAG & INVALIDATION")
    print("=================================================================")

    r = get_redis_client()
    assert r is not None, "REDIS_DSN client is None — refusing to silently skip!"

    req_id = str(uuid.uuid4())
    test_sha = hashlib.sha256(f"test_tag_{req_id}".encode()).hexdigest()
    isolated_doc = f"TEST-{req_id[:8].upper()}/2026/NĐ-CP"
    test_chunk = dict(_fetch_real_chunk(), doc_number=isolated_doc)

    SetCacheWithDocTagsNode({
        "request_id": req_id,
        "query": "Lưu trữ hồ sơ quy hoạch (isolated test)",
        "query_sha256": test_sha,
        "answer": _SHARED.get("answer", "Nội dung IRAC thử nghiệm [Nguồn 1: Điều 21]"),
        "top_chunks": [test_chunk],
        "citations": [test_chunk],
    })

    ckey = cache_key_for_sha(test_sha)
    tkey = doc_tag_key(isolated_doc)
    cached_raw = r.get(ckey)
    assert cached_raw is not None, f"Cache key {ckey} missing in Redis!"
    ttl = r.ttl(ckey)
    tagged_members = r.smembers(tkey) or set()
    assert 86000 <= ttl <= 86400
    assert test_sha in tagged_members

    inv_count = invalidate_by_doc(isolated_doc)
    assert inv_count == 1
    assert r.get(ckey) is None
    assert r.exists(tkey) == 0

    print(f"[PASSED] Isolated Doc-Tag Cache & Invalidation: doc={isolated_doc}, ttl={ttl}s, invalidated={inv_count}, post_get=None\n")


def test_abstain_sse_and_negative_cache_and_postgres_audit():
    print("=================================================================")
    print("TEST 5: FULL GRAPH ABSTAIN SSE + NEGATIVE CACHE + NEON PG AUDIT")
    print("=================================================================")

    async def _run():
        r = get_redis_client()
        assert r is not None, "REDIS_DSN client is None!"
        graph = _build_full_output_graph()

        # A) Ungrounded Abstain through full StateGraph (verifies SSE metadata + refusal token + done + neg cache + PG audit)
        abs_req_id = str(uuid.uuid4())
        neg_sha = hashlib.sha256(f"test_neg_{abs_req_id}".encode()).hexdigest()
        abs_events = []
        async for ev in graph.astream(
            {
                "request_id": abs_req_id,
                "query": "Cách hack mật khẩu wifi theo luật?",
                "cleaned_query": "Cách hack mật khẩu wifi theo luật?",
                "query_sha256": neg_sha,
                "abstained": True,
                "abstain_reason": "out_of_scope",
                "execution_mode": "slow_react_abstain",
                "react_step": 1,
                "start_time": time.time(),
                "latency_ms": {"Qwen7BJsonPlannerNode": 910.0},
            },
            stream_mode="custom",
        ):
            abs_events.append(ev)

        assert len(abs_events) == 3, f"Expected [metadata, token, done], got {abs_events}"
        assert abs_events[0]["event"] == "metadata" and abs_events[0]["data"]["abstained"] is True
        assert abs_events[1]["event"] == "token" and "Hệ thống từ chối trả lời" in abs_events[1]["data"]["delta"]
        assert abs_events[2]["event"] == "done"

        neg_ckey = cache_key_for_sha(neg_sha)
        neg_ttl = r.ttl(neg_ckey)
        assert 500 <= neg_ttl <= 600
        assert neg_sha in (r.smembers(NEGATIVE_CACHE_TAG_KEY) or set())
        r.delete(neg_ckey)
        r.srem(NEGATIVE_CACHE_TAG_KEY, neg_sha)
        print(f"[PASSED] Full-Graph Abstain SSE + Negative Cache: events={[e['event'] for e in abs_events]}, neg_ttl={neg_ttl}s")
        print(f"       Refusal Token: {abs_events[1]['data']['delta']}")

        # B) Grounded Audit Row with real chunk UUID in query_log & query_retrieved_chunk
        real_chunk = _fetch_real_chunk()
        pos_req_id = str(uuid.uuid4())
        t0 = time.time()
        await WriteAuditLogToPostgresNode({
            "request_id": pos_req_id,
            "query": "Quy định về lưu trữ hồ sơ quy hoạch xây dựng theo Điều 21 Luật Xây dựng?",
            "cleaned_query": "Quy định về lưu trữ hồ sơ quy hoạch xây dựng theo Điều 21 Luật Xây dựng?",
            "execution_mode": "fast_path",
            "react_step": 0,
            "doc_filters": ["50/2014/QH13"],
            "article_filter": "Điều 21",
            "cache_hit": False,
            "abstained": False,
            "answer": _SHARED.get("answer", "[Nguồn 1: Điều 21 Văn bản 50/2014/QH13]"),
            "top_chunks": [real_chunk],
            "latency_ms": {"DirectPointGetNode": 15.2, "Top10PruneAndInt8RerankNode": 410.6, "IRACGenerateNode": 3200.0},
        })
        dispatch_ms = (time.time() - t0) * 1000.0
        assert dispatch_ms < 25.0, f"WriteAuditLogToPostgresNode blocked for {dispatch_ms:.2f}ms!"

        await flush_pending_audit_tasks()

        conn = psycopg2.connect(os.getenv("DATABASE_DSN"))
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT request_id::text, execution_mode, react_steps, extracted_doc_number,
                       extracted_article, abstained, retrieval_ms, rerank_ms, generation_ms, llm_model_name
                FROM query_log WHERE request_id = %s::uuid
                """,
                (pos_req_id,),
            )
            pos_row = cur.fetchone()
            cur.execute(
                "SELECT execution_mode, react_steps, abstained FROM query_log WHERE request_id = %s::uuid",
                (abs_req_id,),
            )
            abs_row = cur.fetchone()
            cur.execute(
                "SELECT chunk_id::text, rank_order, rrf_score, rerank_score, passed_threshold "
                "FROM query_retrieved_chunk WHERE request_id = %s::uuid",
                (pos_req_id,),
            )
            chunk_rows = cur.fetchall()
            cur.execute("DELETE FROM query_logs WHERE request_id IN (%s::uuid, %s::uuid)", (pos_req_id, abs_req_id))
        conn.close()

        assert pos_row is not None and pos_row[6] == 15 and pos_row[7] == 411 and pos_row[8] == 3200
        assert abs_row == ("slow_react_abstain", 1, True)
        assert len(chunk_rows) == 1 and chunk_rows[0][4] is True

        print(f"[PASSED] Async Neon PG Audit (dispatch={dispatch_ms:.2f}ms):")
        print(f"       Grounded query_log : {pos_row}")
        print(f"       Abstain  query_log : {abs_row}")
        print(f"       Retrieved chunk row: {chunk_rows[0]}\n")

    asyncio.run(_run())


if __name__ == "__main__":
    test_confidence_gate_routing_matrix()
    test_inventory_table_node_and_context_assembly()
    test_live_irac_sse_streaming()
    test_redis_positive_tagged_caching_and_invalidation()
    test_abstain_sse_and_negative_cache_and_postgres_audit()
    print("ALL CHUNK 5 TESTS COMPLETED SUCCESSFULLY!")
