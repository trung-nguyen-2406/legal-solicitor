import asyncio
import os
import sys
import time
from typing import Any, Dict

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from apps.workflow.nodes_react import (
    DedupQuotaAndAmendmentPinNode,
    FinalChunkBankInt8RerankNode,
    Qwen7BJsonPlannerNode,
    ToolGraphHopNode,
    ToolRewrittenSearchNode,
    route_escalation_gate,
    route_planner_action,
)
from apps.workflow.state import LegalRAGState

_SHARED: Dict[str, Any] = {}
FOREIGN_QUERY = "What is the penalty for running a red light in Vietnam traffic law?"


def test_escalation_gate_rules():
    print("=================================================================")
    print("TEST 1: ESCALATION GATE ROUTING MATRIX")
    print("=================================================================")
    scenarios = [
        (
            "Direct point with chunks",
            {"fast_branch": "direct_point", "top_chunks": [{"chunk_id": "c1"}], "max_score": 0.35},
            "keep_fast_path",
        ),
        (
            "Aggregation with >= 3 chunks and VN",
            {"fast_branch": "aggregation", "top_chunks": [{"chunk_id": f"c{i}"} for i in range(4)], "is_foreign_lang": False, "max_score": 0.38},
            "keep_fast_path",
        ),
        (
            "Low rerank score (<0.42)",
            {"fast_branch": "standard_hybrid", "top_chunks": [{"chunk_id": "c1"}], "max_score": 0.39, "word_count": 20},
            "enter_react",
        ),
        (
            "Foreign language query",
            {"fast_branch": "standard_hybrid", "top_chunks": [{"chunk_id": "c1"}], "max_score": 0.85, "is_foreign_lang": True},
            "enter_react",
        ),
        (
            "Multi-step query",
            {"fast_branch": "standard_hybrid", "top_chunks": [{"chunk_id": "c1"}], "max_score": 0.88, "is_multistep": True},
            "enter_react",
        ),
        (
            "Long word count (>65 words)",
            {"fast_branch": "standard_hybrid", "top_chunks": [{"chunk_id": "c1"}], "max_score": 0.88, "word_count": 72},
            "enter_react",
        ),
    ]

    for label, state_input, expected in scenarios:
        decision = route_escalation_gate(state_input)
        assert decision == expected, f"{label}: got {decision}, expected {expected}"
        print(f"[PASSED] {label} -> {decision}")
    print()


def test_circuit_breakers():
    print("=================================================================")
    print("TEST 2: CIRCUIT BREAKERS (MAX STEPS & TIMEOUT)")
    print("=================================================================")

    async def _run():
        s_step = {"react_step": 2, "start_time": time.time(), "query": "Test"}
        res_step = await Qwen7BJsonPlannerNode(s_step)
        assert res_step["planner_decision"]["action"] == "FINISH"
        assert res_step["execution_mode"] == "slow_react_timeout"
        print(f"[PASSED] Max steps breaker: {res_step['planner_decision']} | mode={res_step['execution_mode']}")

        s_time = {"react_step": 0, "start_time": time.time() - 36.0, "query": "Test"}
        res_time = await Qwen7BJsonPlannerNode(s_time)
        assert res_time["planner_decision"]["action"] == "FINISH"
        assert res_time["execution_mode"] == "slow_react_timeout"
        print(f"[PASSED] Timeout breaker  : {res_time['planner_decision']} | mode={res_time['execution_mode']}\n")

    asyncio.run(_run())


def test_ollama_planner_decisions():
    print("=================================================================")
    print("TEST 3: LIVE OLLAMA QWEN2.5-7B JSON PLANNER")
    print("=================================================================")

    async def _run():
        out_of_scope_query = "Who invented the light bulb and what year?"
        t0 = time.time()
        res_abs = await Qwen7BJsonPlannerNode({
            "query": out_of_scope_query,
            "pointer_scratchpad": [],
            "react_step": 0,
            "start_time": time.time(),
        })
        abs_dur = time.time() - t0
        dec_abs = res_abs["planner_decision"]
        assert dec_abs["action"] == "ABSTAIN"
        assert route_planner_action(res_abs) == "abstain"
        print(f"[PASSED] Out-of-scope ({abs_dur:.2f}s): action={dec_abs['action']} reason={dec_abs.get('abstain_reason')}")
        print(f"       Thought: {dec_abs.get('thought')}")

        t1 = time.time()
        res_for = await Qwen7BJsonPlannerNode({
            "query": FOREIGN_QUERY,
            "pointer_scratchpad": [],
            "react_step": 0,
            "start_time": time.time(),
        })
        for_dur = time.time() - t1
        dec_for = res_for["planner_decision"]
        search_args = dec_for.get("search_args") or {}
        assert dec_for["action"] == "tool_rewritten_search"
        assert route_planner_action(res_for) == "rewritten_search"
        assert len(search_args.get("legal_queries") or []) >= 1
        print(f"[PASSED] Foreign query ({for_dur:.2f}s): action={dec_for['action']}")
        print(f"       Search Args: {search_args}\n")
        _SHARED["planner_decision"] = dec_for

    asyncio.run(_run())


def test_tool_rewritten_search():
    print("=================================================================")
    print("TEST 4: TOOL REWRITTEN SEARCH (HYBRID RRF + METADATA HYDRATION)")
    print("=================================================================")
    planner_decision = _SHARED.get("planner_decision") or {
        "action": "tool_rewritten_search",
        "search_args": {
            "legal_queries": ["Phạt vi phạm tín hiệu đèn giao thông theo luật giao thông đường bộ Việt Nam"],
            "year_from": None,
            "year_to": None,
            "legal_field": "Giao thông",
        },
    }

    t0 = time.time()
    search_out = ToolRewrittenSearchNode({
        "query": FOREIGN_QUERY,
        "planner_decision": planner_decision,
        "pointer_scratchpad": [],
        "matched_subqueries": {},
        "chunk_bank": {},
    })
    dur = time.time() - t0

    bank = search_out["chunk_bank"]
    scratchpad = search_out["pointer_scratchpad"]
    matched = search_out["matched_subqueries"]

    assert len(bank) >= 5
    assert len(scratchpad) >= 1
    assert all(float(c.get("rerank_score", 0.0)) == 0.0 for c in bank.values()), "Un-reranked chunks must keep rerank_score=0.0!"
    assert all("Điều" in (p.get("article_code") or "") or "ALL" in (p.get("article_code") or "") for p in scratchpad)

    print(f"[PASSED] ToolRewrittenSearchNode ({dur:.2f}s): bank={len(bank)}, scratchpad={len(scratchpad)}, matched={len(matched)}")
    for idx, p in enumerate(scratchpad[:3], start=1):
        print(f"       {idx}. [{p.get('doc_number')}:{p.get('article_code')}] rrf={p.get('score'):.4f} | {p.get('title_lead_in')[:85]}...")
    print()
    _SHARED["search_out"] = search_out


def test_tool_graph_hop():
    print("=================================================================")
    print("TEST 5: TOOL GRAPH HOP (POSTGRESQL RELATIONS & PROVISION CITATIONS)")
    print("=================================================================")
    t0 = time.time()
    hop_out = ToolGraphHopNode({
        "planner_decision": {
            "thought": "Need related provisions and citing laws for 50/2014/QH13",
            "action": "tool_graph_hop",
            "graph_args": {"doc_number": "50/2014/QH13", "article_code": None, "direction": "all"},
        },
        "chunk_bank": {},
        "pointer_scratchpad": [],
    })
    dur = time.time() - t0

    bank = hop_out["chunk_bank"]
    scratchpad = hop_out["pointer_scratchpad"]
    assert len(bank) > 0
    assert len(scratchpad) > 0
    assert all(c.get("doc_number") != "50/2014/QH13" for c in bank.values()), "Graph hop returned self-document chunks!"
    assert all(float(c.get("rerank_score", 0.0)) == 0.0 for c in bank.values())

    print(f"[PASSED] ToolGraphHopNode ({dur:.2f}s): hopped={len(bank)} external chunks")
    for idx, p in enumerate(scratchpad[:3], start=1):
        print(f"       {idx}. [{p.get('doc_number')}:{p.get('article_code')}] prior={p.get('score')} | {p.get('title_lead_in')[:85]}...")
    print()
    _SHARED["hop_out"] = hop_out


def test_final_rerank_and_dedup_hygiene():
    print("=================================================================")
    print("TEST 6: FINAL INT8 RERANK (FOREIGN QUERY + SUBQUERY RESCUE) & DEDUP")
    print("=================================================================")
    if "search_out" not in _SHARED:
        test_tool_rewritten_search()
    if "hop_out" not in _SHARED:
        test_tool_graph_hop()

    combined_bank = dict(_SHARED["search_out"]["chunk_bank"])
    combined_bank.update(_SHARED["hop_out"]["chunk_bank"])

    # Keep the REAL English query in cleaned_query to prove matched_subqueries rescues cross-lingual dilution
    no_subq_out = FinalChunkBankInt8RerankNode({
        "query": FOREIGN_QUERY,
        "cleaned_query": FOREIGN_QUERY,
        "chunk_bank": combined_bank,
        "matched_subqueries": {},
    })

    state: LegalRAGState = {
        "query": FOREIGN_QUERY,
        "cleaned_query": FOREIGN_QUERY,
        "chunk_bank": combined_bank,
        "matched_subqueries": _SHARED["search_out"]["matched_subqueries"],
        "doc_filters": [],
    }
    t0 = time.time()
    rerank_out = FinalChunkBankInt8RerankNode(state)
    dur_rerank = time.time() - t0

    cands = rerank_out["rrf_candidates"]
    logits = rerank_out["raw_logits"]
    assert len(cands) <= 20 and len(cands) == len(logits)
    assert max(logits) > max(no_subq_out["raw_logits"]), "matched_subqueries failed to boost max logit over raw English query!"

    state["rrf_candidates"] = cands
    state["raw_logits"] = logits
    dedup_out = DedupQuotaAndAmendmentPinNode(state)
    top_chunks = dedup_out["top_chunks"]
    max_score = dedup_out["max_score"]

    assert 1 <= len(top_chunks) <= 7
    assert max_score >= 0.85
    assert set(dedup_out["chunk_bank"].keys()) == {c["chunk_id"] for c in top_chunks}

    print(
        f"[PASSED] Subquery rescue verified ({dur_rerank:.2f}s): "
        f"English-only max_logit={max(no_subq_out['raw_logits']):.3f} -> "
        f"Subquery-rescued max_logit={max(logits):.3f} (calibrated max_score={max_score:.4f})"
    )
    for rank, c in enumerate(top_chunks, start=1):
        pin_flag = "[PINNED AMENDMENT]" if c.get("is_pinned_amendment") else ""
        tag = c.get("anchor_tag") or f"[{c.get('doc_number', '')}:{c.get('article_code', '')}]"
        snippet = (c.get("raw_content") or c.get("breadcrumb") or "").replace("\n", " ")[:100]
        print(f"       #{rank} {tag} Score={c.get('rerank_score', 0.0):.4f} (Year={c.get('issued_year')}) {pin_flag}")
        print(f"          {snippet}...")


if __name__ == "__main__":
    test_escalation_gate_rules()
    test_circuit_breakers()
    test_ollama_planner_decisions()
    test_tool_rewritten_search()
    test_tool_graph_hop()
    test_final_rerank_and_dedup_hygiene()
    print("\nALL CHUNK 4 TESTS COMPLETED SUCCESSFULLY!")
