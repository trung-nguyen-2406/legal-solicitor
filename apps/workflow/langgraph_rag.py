import time
from typing import Literal, Optional
import uuid
from langgraph.graph import END, START, StateGraph

from apps.services.resources import get_descriptor_service
from apps.workflow.nodes_entry import (
    AbstainFastExitNode,
    EncodeQueryAndLinearDomainProbeNode,
    PureRedisCacheCheckNode,
    ReturnCachedResponseNode,
    StructuredQueryDescriptorNode,
    WriteShortTtlNegativeCacheNode,
    route_after_cache_check,
    route_after_domain_probe,
)
from apps.workflow.nodes_fast import (
    BiDirectionalGraphExpandNode,
    DedupQuotaAndTemporalBoostNode,
    DirectPointGetNode,
    FilteredSemanticOrLineageAggNode,
    ScopedHybridRrfPrefetch50Node,
    StratifiedMinQuotaSearchNode,
    Top10PruneAndInt8RerankNode,
    route_fast_path_branch,
)
from apps.workflow.nodes_output import (
    AbstainWithReasonNode,
    ConfidenceGate,
    GenerateHierarchicalInventoryTableNode,
    IRACGenerateNode,
    ReturnResponseNode,
    SetCacheWithDocTagsNode,
    SetGlobalNegativeCacheNode,
    WriteAuditLogToPostgresNode,
    route_after_confidence,
)
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

EXPECTED_NODES = (
    "PureRedisCacheCheckNode",
    "ReturnCachedResponseNode",
    "StructuredQueryDescriptorNode",
    "EncodeQueryAndLinearDomainProbeNode",
    "AbstainFastExitNode",
    "WriteShortTtlNegativeCacheNode",
    "DirectPointGetNode",
    "BiDirectionalGraphExpandNode",
    "FilteredSemanticOrLineageAggNode",
    "StratifiedMinQuotaSearchNode",
    "ScopedHybridRrfPrefetch50Node",
    "Top10PruneAndInt8RerankNode",
    "DedupQuotaAndTemporalBoostNode",
    "Qwen7BJsonPlannerNode",
    "ToolRewrittenSearchNode",
    "ToolGraphHopNode",
    "FinalChunkBankInt8RerankNode",
    "DedupQuotaAndAmendmentPinNode",
    "ConfidenceGate",
    "GenerateHierarchicalInventoryTableNode",
    "IRACGenerateNode",
    "SetCacheWithDocTagsNode",
    "AbstainWithReasonNode",
    "SetGlobalNegativeCacheNode",
    "WriteAuditLogToPostgresNode",
    "ReturnResponseNode",
)


def route_domain_or_fast_branch(
    state: LegalRAGState,
) -> Literal["fast_abstain", "direct_point", "aggregation", "stratified_multi", "standard_hybrid"]:
    if route_after_domain_probe(state) == "fast_abstain":
        return "fast_abstain"
    return route_fast_path_branch(state)


def build_initial_state(query: str, request_id: Optional[str] = None) -> LegalRAGState:
    req_id = request_id or str(uuid.uuid4())
    parsed = get_descriptor_service().parse(query)
    return {
        "request_id": req_id,
        "query": query,
        "query_sha256": parsed["query_sha256"],
        "start_time": time.time(),
        "react_step": 0,
        "pointer_scratchpad": [],
        "chunk_bank": {},
        "latency_ms": {},
    }


def build_legal_rag_workflow() -> StateGraph:
    workflow = StateGraph(LegalRAGState)

    workflow.add_node("PureRedisCacheCheckNode", PureRedisCacheCheckNode)
    workflow.add_node("ReturnCachedResponseNode", ReturnCachedResponseNode)
    workflow.add_node("StructuredQueryDescriptorNode", StructuredQueryDescriptorNode)
    workflow.add_node("EncodeQueryAndLinearDomainProbeNode", EncodeQueryAndLinearDomainProbeNode)
    workflow.add_node("AbstainFastExitNode", AbstainFastExitNode)
    workflow.add_node("WriteShortTtlNegativeCacheNode", WriteShortTtlNegativeCacheNode)

    workflow.add_node("DirectPointGetNode", DirectPointGetNode)
    workflow.add_node("BiDirectionalGraphExpandNode", BiDirectionalGraphExpandNode)
    workflow.add_node("FilteredSemanticOrLineageAggNode", FilteredSemanticOrLineageAggNode)
    workflow.add_node("StratifiedMinQuotaSearchNode", StratifiedMinQuotaSearchNode)
    workflow.add_node("ScopedHybridRrfPrefetch50Node", ScopedHybridRrfPrefetch50Node)
    workflow.add_node("Top10PruneAndInt8RerankNode", Top10PruneAndInt8RerankNode)
    workflow.add_node("DedupQuotaAndTemporalBoostNode", DedupQuotaAndTemporalBoostNode)

    workflow.add_node("Qwen7BJsonPlannerNode", Qwen7BJsonPlannerNode)
    workflow.add_node("ToolRewrittenSearchNode", ToolRewrittenSearchNode)
    workflow.add_node("ToolGraphHopNode", ToolGraphHopNode)
    workflow.add_node("FinalChunkBankInt8RerankNode", FinalChunkBankInt8RerankNode)
    workflow.add_node("DedupQuotaAndAmendmentPinNode", DedupQuotaAndAmendmentPinNode)

    workflow.add_node("ConfidenceGate", ConfidenceGate)
    workflow.add_node("GenerateHierarchicalInventoryTableNode", GenerateHierarchicalInventoryTableNode)
    workflow.add_node("IRACGenerateNode", IRACGenerateNode)
    workflow.add_node("SetCacheWithDocTagsNode", SetCacheWithDocTagsNode)
    workflow.add_node("AbstainWithReasonNode", AbstainWithReasonNode)
    workflow.add_node("SetGlobalNegativeCacheNode", SetGlobalNegativeCacheNode)
    workflow.add_node("WriteAuditLogToPostgresNode", WriteAuditLogToPostgresNode)
    workflow.add_node("ReturnResponseNode", ReturnResponseNode)

    workflow.add_edge(START, "PureRedisCacheCheckNode")
    workflow.add_conditional_edges(
        "PureRedisCacheCheckNode",
        route_after_cache_check,
        {
            "cache_hit": "ReturnCachedResponseNode",
            "cache_miss": "StructuredQueryDescriptorNode",
        },
    )
    workflow.add_edge("ReturnCachedResponseNode", END)

    workflow.add_edge("StructuredQueryDescriptorNode", "EncodeQueryAndLinearDomainProbeNode")
    workflow.add_conditional_edges(
        "EncodeQueryAndLinearDomainProbeNode",
        route_domain_or_fast_branch,
        {
            "fast_abstain": "AbstainFastExitNode",
            "direct_point": "DirectPointGetNode",
            "aggregation": "FilteredSemanticOrLineageAggNode",
            "stratified_multi": "StratifiedMinQuotaSearchNode",
            "standard_hybrid": "ScopedHybridRrfPrefetch50Node",
        },
    )

    workflow.add_edge("AbstainFastExitNode", "WriteShortTtlNegativeCacheNode")
    workflow.add_edge("WriteShortTtlNegativeCacheNode", "WriteAuditLogToPostgresNode")

    workflow.add_edge("DirectPointGetNode", "BiDirectionalGraphExpandNode")
    workflow.add_edge("StratifiedMinQuotaSearchNode", "Top10PruneAndInt8RerankNode")
    workflow.add_edge("ScopedHybridRrfPrefetch50Node", "Top10PruneAndInt8RerankNode")
    workflow.add_edge("Top10PruneAndInt8RerankNode", "DedupQuotaAndTemporalBoostNode")

    escalation_map = {
        "keep_fast_path": "ConfidenceGate",
        "enter_react": "Qwen7BJsonPlannerNode",
    }
    workflow.add_conditional_edges("BiDirectionalGraphExpandNode", route_escalation_gate, escalation_map)
    workflow.add_conditional_edges("FilteredSemanticOrLineageAggNode", route_escalation_gate, escalation_map)
    workflow.add_conditional_edges("DedupQuotaAndTemporalBoostNode", route_escalation_gate, escalation_map)

    workflow.add_conditional_edges(
        "Qwen7BJsonPlannerNode",
        route_planner_action,
        {
            "abstain": "AbstainWithReasonNode",
            "rewritten_search": "ToolRewrittenSearchNode",
            "graph_hop": "ToolGraphHopNode",
            "finish_react": "FinalChunkBankInt8RerankNode",
        },
    )
    workflow.add_edge("ToolRewrittenSearchNode", "Qwen7BJsonPlannerNode")
    workflow.add_edge("ToolGraphHopNode", "Qwen7BJsonPlannerNode")
    workflow.add_edge("FinalChunkBankInt8RerankNode", "DedupQuotaAndAmendmentPinNode")
    workflow.add_edge("DedupQuotaAndAmendmentPinNode", "ConfidenceGate")

    workflow.add_conditional_edges(
        "ConfidenceGate",
        route_after_confidence,
        {
            "grounded_inventory": "GenerateHierarchicalInventoryTableNode",
            "grounded_direct_irac": "IRACGenerateNode",
            "ungrounded_abstain": "AbstainWithReasonNode",
        },
    )
    workflow.add_edge("GenerateHierarchicalInventoryTableNode", "IRACGenerateNode")
    workflow.add_edge("IRACGenerateNode", "SetCacheWithDocTagsNode")
    workflow.add_edge("SetCacheWithDocTagsNode", "WriteAuditLogToPostgresNode")

    workflow.add_edge("AbstainWithReasonNode", "SetGlobalNegativeCacheNode")
    workflow.add_edge("SetGlobalNegativeCacheNode", "WriteAuditLogToPostgresNode")

    workflow.add_edge("WriteAuditLogToPostgresNode", "ReturnResponseNode")
    workflow.add_edge("ReturnResponseNode", END)

    return workflow


workflow = build_legal_rag_workflow()
compiled_graph = workflow.compile()

