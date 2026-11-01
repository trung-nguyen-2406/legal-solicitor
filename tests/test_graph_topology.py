from collections import defaultdict, deque
import sys
from typing import Dict, Set

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from apps.workflow.langgraph_rag import EXPECTED_NODES, compiled_graph


def test_graph_mermaid_and_registered_nodes():
    drawable = compiled_graph.get_graph()
    mermaid_str = drawable.draw_mermaid()

    registered_user_nodes = {k for k in compiled_graph.nodes.keys() if not k.startswith("__")}
    expected_set = set(EXPECTED_NODES)

    missing = expected_set - registered_user_nodes
    unexpected = registered_user_nodes - expected_set

    assert not missing, f"Missing expected SDD nodes: {sorted(missing)}"
    assert not unexpected, f"Unexpected extra nodes registered: {sorted(unexpected)}"
    assert "PureRedisCacheCheckNode" in mermaid_str
    assert "Qwen7BJsonPlannerNode" in mermaid_str
    assert "IRACGenerateNode" in mermaid_str

    print("=================================================================")
    print("COMPILED LANGGRAPH MERMAID TOPOLOGY (SDD v4.0 Section 3.4.3)")
    print("=================================================================")
    print(mermaid_str)
    print(f"[PASSED] Registered all {len(registered_user_nodes)} SDD nodes cleanly.\n")


def test_no_orphan_or_dead_end_nodes_and_required_edges():
    drawable = compiled_graph.get_graph()
    all_node_ids = set(drawable.nodes.keys())

    adj: Dict[str, Set[str]] = defaultdict(set)
    rev_adj: Dict[str, Set[str]] = defaultdict(set)
    edge_pairs = set()

    for edge in drawable.edges:
        src = str(edge.source)
        dst = str(edge.target)
        adj[src].add(dst)
        rev_adj[dst].add(src)
        edge_pairs.add((src, dst))

    assert "__start__" in all_node_ids, "Missing __start__ node"
    assert "__end__" in all_node_ids, "Missing __end__ node"

    # 1. Forward BFS from __start__: every node must be reachable
    visited_from_start: Set[str] = set()
    q_fwd = deque(["__start__"])
    while q_fwd:
        curr = q_fwd.popleft()
        if curr in visited_from_start:
            continue
        visited_from_start.add(curr)
        for nxt in adj.get(curr, set()):
            if nxt not in visited_from_start:
                q_fwd.append(nxt)

    unreachable_from_start = all_node_ids - visited_from_start
    assert not unreachable_from_start, f"Orphan nodes unreachable from START: {sorted(unreachable_from_start)}"

    # 2. Backward BFS from __end__: every node must have a valid path to END (no dead ends / inescapable cycles)
    can_reach_end: Set[str] = set()
    q_rev = deque(["__end__"])
    while q_rev:
        curr = q_rev.popleft()
        if curr in can_reach_end:
            continue
        can_reach_end.add(curr)
        for prev in rev_adj.get(curr, set()):
            if prev not in can_reach_end:
                q_rev.append(prev)

    dead_end_nodes = all_node_ids - can_reach_end
    assert not dead_end_nodes, f"Dead-end nodes that cannot reach END: {sorted(dead_end_nodes)}"

    # 3. Verify critical SDD v4.0 Section 3.4.3 transitions explicitly
    required_edges = [
        ("__start__", "PureRedisCacheCheckNode"),
        ("PureRedisCacheCheckNode", "ReturnCachedResponseNode"),
        ("PureRedisCacheCheckNode", "StructuredQueryDescriptorNode"),
        ("ReturnCachedResponseNode", "__end__"),
        ("StructuredQueryDescriptorNode", "EncodeQueryAndLinearDomainProbeNode"),
        ("EncodeQueryAndLinearDomainProbeNode", "AbstainFastExitNode"),
        ("EncodeQueryAndLinearDomainProbeNode", "DirectPointGetNode"),
        ("EncodeQueryAndLinearDomainProbeNode", "FilteredSemanticOrLineageAggNode"),
        ("EncodeQueryAndLinearDomainProbeNode", "StratifiedMinQuotaSearchNode"),
        ("EncodeQueryAndLinearDomainProbeNode", "ScopedHybridRrfPrefetch50Node"),
        ("AbstainFastExitNode", "WriteShortTtlNegativeCacheNode"),
        ("WriteShortTtlNegativeCacheNode", "WriteAuditLogToPostgresNode"),
        ("DirectPointGetNode", "BiDirectionalGraphExpandNode"),
        ("StratifiedMinQuotaSearchNode", "Top10PruneAndInt8RerankNode"),
        ("ScopedHybridRrfPrefetch50Node", "Top10PruneAndInt8RerankNode"),
        ("Top10PruneAndInt8RerankNode", "DedupQuotaAndTemporalBoostNode"),
        ("BiDirectionalGraphExpandNode", "ConfidenceGate"),
        ("BiDirectionalGraphExpandNode", "Qwen7BJsonPlannerNode"),
        ("FilteredSemanticOrLineageAggNode", "ConfidenceGate"),
        ("FilteredSemanticOrLineageAggNode", "Qwen7BJsonPlannerNode"),
        ("DedupQuotaAndTemporalBoostNode", "ConfidenceGate"),
        ("DedupQuotaAndTemporalBoostNode", "Qwen7BJsonPlannerNode"),
        ("Qwen7BJsonPlannerNode", "AbstainWithReasonNode"),
        ("Qwen7BJsonPlannerNode", "ToolRewrittenSearchNode"),
        ("Qwen7BJsonPlannerNode", "ToolGraphHopNode"),
        ("Qwen7BJsonPlannerNode", "FinalChunkBankInt8RerankNode"),
        ("ToolRewrittenSearchNode", "Qwen7BJsonPlannerNode"),
        ("ToolGraphHopNode", "Qwen7BJsonPlannerNode"),
        ("FinalChunkBankInt8RerankNode", "DedupQuotaAndAmendmentPinNode"),
        ("DedupQuotaAndAmendmentPinNode", "ConfidenceGate"),
        ("ConfidenceGate", "GenerateHierarchicalInventoryTableNode"),
        ("ConfidenceGate", "IRACGenerateNode"),
        ("ConfidenceGate", "AbstainWithReasonNode"),
        ("GenerateHierarchicalInventoryTableNode", "IRACGenerateNode"),
        ("IRACGenerateNode", "SetCacheWithDocTagsNode"),
        ("SetCacheWithDocTagsNode", "WriteAuditLogToPostgresNode"),
        ("AbstainWithReasonNode", "SetGlobalNegativeCacheNode"),
        ("SetGlobalNegativeCacheNode", "WriteAuditLogToPostgresNode"),
        ("WriteAuditLogToPostgresNode", "ReturnResponseNode"),
        ("ReturnResponseNode", "__end__"),
    ]
    missing_edges = [e for e in required_edges if e not in edge_pairs]
    assert not missing_edges, f"Missing required SDD edges: {missing_edges}"

    print("=================================================================")
    print("GRAPH CONNECTIVITY & DEAD-END AUDIT")
    print("=================================================================")
    print(f"[PASSED] Total graph vertices (incl. __start__/__end__): {len(all_node_ids)}")
    print(f"[PASSED] Total directed/conditional edges             : {len(drawable.edges)}")
    print(f"[PASSED] Reachable from START                         : {len(visited_from_start)}/{len(all_node_ids)} (0 orphans)")
    print(f"[PASSED] Path to END verified                         : {len(can_reach_end)}/{len(all_node_ids)} (0 dead-ends)")
    print(f"[PASSED] Verified all {len(required_edges)} SDD v4.0 Section 3.4.3 transitions.\n")


if __name__ == "__main__":
    test_graph_mermaid_and_registered_nodes()
    test_no_orphan_or_dead_end_nodes_and_required_edges()
    print("ALL CHUNK 6 GRAPH TOPOLOGY CHECKS PASSED!")

