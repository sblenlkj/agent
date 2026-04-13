from __future__ import annotations

from loguru import logger

from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentRunState
from src.agent.nodes.n1_prepare_input_bundle_node import prepare_input_bundle_node, prepare_input_bundle_from_cache_node
from src.agent.nodes.n2_plan_joins_node import plan_joins_node
from src.agent.nodes.n3_generate_feature_ideas_node import generate_feature_ideas_node
from src.agent.nodes.n4_generate_and_apply_feature_code_node import (
    generate_and_apply_feature_code_node,
)
from src.agent.nodes.n5_apply_legacy_executor_node import (
    apply_legacy_executor_node,
)
from src.agent.nodes.n6_train_catboost_and_select_top_features_node import (
    train_catboost_and_select_top_features_node,
)

def build_agent_graph(input_from_cache: bool = False):
    graph = StateGraph(AgentRunState)

    if input_from_cache:
        logger.warning("Cache for input bundle!!!")
        graph.add_node("prepare_input_bundle", prepare_input_bundle_from_cache_node)
    else:
        graph.add_node("prepare_input_bundle", prepare_input_bundle_node)

    graph.add_node("plan_joins", plan_joins_node)
    graph.add_node("generate_feature_ideas", generate_feature_ideas_node)
    graph.add_node(
        "generate_and_apply_feature_code",
        generate_and_apply_feature_code_node,
    )
    graph.add_node("apply_legacy_executor", apply_legacy_executor_node)
    graph.add_node(
        "train_catboost_and_select_top_features",
        train_catboost_and_select_top_features_node,
    )

    graph.add_edge(START, "prepare_input_bundle")
    graph.add_edge("prepare_input_bundle", "plan_joins")
    graph.add_edge("plan_joins", "generate_feature_ideas")
    graph.add_edge("generate_feature_ideas", "generate_and_apply_feature_code")
    graph.add_edge("generate_and_apply_feature_code", "apply_legacy_executor")
    graph.add_edge("apply_legacy_executor", END)
    graph.add_edge("apply_legacy_executor", "train_catboost_and_select_top_features")
    graph.add_edge("train_catboost_and_select_top_features", END)

    return graph.compile()

