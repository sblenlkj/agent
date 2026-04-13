from __future__ import annotations

from typing import TypedDict

from src.common.feature_codegen.models import GeneratedFeatureCode
from src.common.feature_ideas_generation.models import FeatureIdea
from src.common.io.models import InputBundle
from src.common.joins.planner_module.models import JoinCandidate, JoinPlan


class AgentRunState(TypedDict, total=False):
    input_bundle: InputBundle
    join_candidates: list[JoinCandidate]
    join_plan: JoinPlan
    feature_ideas: list[FeatureIdea]
    generated_feature_codes: list[GeneratedFeatureCode]
    applied_feature_titles: list[str]
    selected_feature_names: list[str]
    validation_auc_all_features: float
    validation_auc_top5_features: float