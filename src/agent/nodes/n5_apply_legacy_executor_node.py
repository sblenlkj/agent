from __future__ import annotations

from loguru import logger

from src.agent.runtime import get_agent_globals
from src.agent.state import AgentRunState
from src.common.runtime import GLOBAL_RUNTIME_BUDGET


def apply_legacy_executor_node(
    state: AgentRunState,
) -> dict[str, object]:
    logger.info("Старт ноды apply_legacy_executor_node")
    GLOBAL_RUNTIME_BUDGET.check(stage_name="apply_legacy_executor_node.start")

    bundle = state.get("input_bundle")
    if bundle is None:
        raise ValueError("В state отсутствует input_bundle")

    join_plan = state.get("join_plan")
    if join_plan is None:
        raise ValueError("В state отсутствует join_plan")

    globals_ = get_agent_globals()
    builder = globals_.generated_features_dataset_builder

    train_df_before = builder.train_df
    test_df_before = builder.test_df

    logger.info(
        "Перед legacy executor: train_shape={}x{}, test_shape={}x{}",
        len(train_df_before),
        len(train_df_before.columns),
        len(test_df_before),
        len(test_df_before.columns),
    )

    table_plans = globals_.legacy_feature_planning_servicer.build_discovered_table_plans(
        bundle=bundle,
        join_plan=join_plan,
    )

    updated_train_df, updated_test_df = globals_.legacy_executor.execute(
        bundle=bundle,
        table_plans=table_plans,
        train_df=globals_.generated_features_dataset_builder.train_df,
        test_df=globals_.generated_features_dataset_builder.test_df,
    )

    if len(updated_train_df) != len(train_df_before):
        raise ValueError(
            "Legacy executor изменил число строк train_df: "
            f"{len(train_df_before)} -> {len(updated_train_df)}"
        )

    if len(updated_test_df) != len(test_df_before):
        raise ValueError(
            "Legacy executor изменил число строк test_df: "
            f"{len(test_df_before)} -> {len(updated_test_df)}"
        )

    added_train_columns = [
        column_name
        for column_name in updated_train_df.columns
        if column_name not in train_df_before.columns
    ]
    added_test_columns = [
        column_name
        for column_name in updated_test_df.columns
        if column_name not in test_df_before.columns
    ]

    added_columns = sorted(set(added_train_columns) | set(added_test_columns))

    builder._train_df = updated_train_df
    builder._test_df = updated_test_df

    logger.info(
        "Нода apply_legacy_executor_node завершена: "
        "train_shape={}x{}, test_shape={}x{}, added_columns={}",
        len(builder.train_df),
        len(builder.train_df.columns),
        len(builder.test_df),
        len(builder.test_df.columns),
        added_columns,
    )

    GLOBAL_RUNTIME_BUDGET.check(stage_name="apply_legacy_executor_node.end")

    return {
        "applied_feature_titles": state.get("applied_feature_titles", []) + added_columns,
    }