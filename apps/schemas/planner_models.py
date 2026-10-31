from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class RewrittenSearchArgs(BaseModel):
    legal_queries: List[str] = Field(..., min_length=1, max_length=3)
    year_from: Optional[int] = Field(None, ge=1945, le=2026)
    year_to: Optional[int] = Field(None, ge=1945, le=2026)
    legal_field: Optional[str] = None


class GraphHopArgs(BaseModel):
    doc_number: str
    article_code: Optional[str] = None
    direction: Literal["cites", "amends", "all"] = "all"


class AggregationSweepArgs(BaseModel):
    sub_queries: List[str] = Field(..., min_length=1, max_length=4)
    anchor_keywords: List[str] = Field(..., min_length=1, max_length=5)
    year_from: Optional[int] = Field(None, ge=1945, le=2026)
    year_to: Optional[int] = Field(None, ge=1945, le=2026)


class PlannerActionOutput(BaseModel):
    thought: str = Field(..., max_length=300)
    action: Literal["ABSTAIN", "tool_rewritten_search", "tool_aggregation_sweep", "tool_graph_hop", "FINISH"]
    abstain_reason: Optional[Literal["out_of_scope", "harmful_or_fictional", "insufficient_evidence"]] = None
    search_args: Optional[RewrittenSearchArgs] = None
    graph_args: Optional[GraphHopArgs] = None
    agg_args: Optional[AggregationSweepArgs] = None

    @model_validator(mode="after")
    def validate_action_arguments(self) -> "PlannerActionOutput":
        if self.action == "tool_rewritten_search" and not self.search_args:
            raise ValueError("search_args required for tool_rewritten_search")
        if self.action == "tool_aggregation_sweep" and not self.agg_args:
            raise ValueError("agg_args required for tool_aggregation_sweep")
        if self.action == "tool_graph_hop" and not self.graph_args:
            raise ValueError("graph_args required for tool_graph_hop")
        if self.action == "ABSTAIN" and not self.abstain_reason:
            raise ValueError("abstain_reason required for ABSTAIN")
        return self
