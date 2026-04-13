from __future__ import annotations

from loguru import logger

from src.agent.runtime import get_agent_globals
from src.agent.state import AgentRunState
from src.common.runtime import GLOBAL_RUNTIME_BUDGET


def train_catboost_and_select_top_features_node(
    state: AgentRunState,
) -> dict[str, object]:
    logger.info("Старт ноды train_catboost_and_select_top_features_node")
    GLOBAL_RUNTIME_BUDGET.check(
        stage_name="train_catboost_and_select_top_features_node.start"
    )

    globals_ = get_agent_globals()
    builder = globals_.generated_features_dataset_builder

    train_df = builder.train_df
    test_df = builder.test_df

    logger.info(
        "Перед обучением CatBoost: train_shape={}x{}, test_shape={}x{}",
        len(train_df),
        len(train_df.columns),
        len(test_df),
        len(test_df.columns),
    )

    result = globals_.catboost_feature_selector.run(
        train_df=train_df,
        test_df=test_df,
    )

    logger.info(
        "Нода train_catboost_and_select_top_features_node завершена: "
        "auc_all={:.6f}, auc_top5={:.6f}, selected_feature_names={}",
        result.validation_auc_all_features,
        result.validation_auc_top5_features,
        result.selected_feature_names,
    )

    GLOBAL_RUNTIME_BUDGET.check(
        stage_name="train_catboost_and_select_top_features_node.end"
    )

    return {
        "selected_feature_names": result.selected_feature_names,
        "validation_auc_all_features": result.validation_auc_all_features,
        "validation_auc_top5_features": result.validation_auc_top5_features,
    }