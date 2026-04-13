from __future__ import annotations

from loguru import logger

from src.agent.runtime import get_agent_globals
from src.agent.state import AgentRunState
from src.common.runtime import GLOBAL_RUNTIME_BUDGET

from src.common.constraints import MAX_FEATURES_GENERATION

def generate_feature_ideas_node(
    state: AgentRunState,
) -> dict[str, object]:
    logger.info("Старт ноды generate_feature_ideas_node")

    GLOBAL_RUNTIME_BUDGET.check(stage_name="generate_feature_ideas_node.start")

    bundle = state.get("input_bundle")
    if bundle is None:
        raise ValueError("В state отсутствует input_bundle")

    join_plan = state.get("join_plan")
    if join_plan is None:
        raise ValueError("В state отсутствует join_plan")

    globals_ = get_agent_globals()

    feature_ideas = globals_.feature_ideas_generation_service.generate(
        bundle=bundle,
        join_plan=join_plan,
        max_feature_ideas=MAX_FEATURES_GENERATION
    )

    logger.info(
        "Нода generate_feature_ideas_node завершена: feature_ideas_count={}",
        len(feature_ideas),
    )

    GLOBAL_RUNTIME_BUDGET.check(stage_name="generate_feature_ideas_node.end")

    return {
        "feature_ideas": feature_ideas,
    }