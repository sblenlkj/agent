from __future__ import annotations

from loguru import logger

from src.agent.state import AgentRunState
from src.common.joins.planner_module.planner import JoinPlanner
from src.common.runtime import GLOBAL_RUNTIME_BUDGET


def plan_joins_node(
    state: AgentRunState,
) -> dict[str, object]:
    logger.info("Старт ноды plan_joins_node")

    GLOBAL_RUNTIME_BUDGET.check(stage_name="plan_joins_node.start")

    bundle = state.get("input_bundle")
    if bundle is None:
        raise ValueError(
            "В state отсутствует input_bundle. "
            "Нельзя построить join plan без prepare_input_bundle_node."
        )

    planner = JoinPlanner()

    logger.info("Строю join candidates")
    candidates = planner.build_candidates(bundle)

    logger.info("Строю join plan")
    plan = planner.build_plan(bundle)

    logger.info(
        "Join planning завершен: candidates_count={}, edges_count={}, skipped_tables={}",
        len(candidates),
        len(plan.edges),
        plan.skipped_tables,
    )

    for edge in plan.edges:
        logger.info(
            "{}.{} -> {}.{} | relation_type={} | requires_aggregation={} | path={}",
            edge.parent_table,
            edge.parent_key,
            edge.child_table,
            edge.child_key,
            edge.relation_type,
            edge.requires_aggregation,
            edge.path_from_train,
        )

    GLOBAL_RUNTIME_BUDGET.check(stage_name="plan_joins_node.end")


    logger.info(candidates)
    logger.info(plan)

    return {
        "join_candidates": candidates,
        "join_plan": plan,
    }