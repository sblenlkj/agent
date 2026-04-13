from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.common.io.file_manager import FileManager
from src.common.io.models import InputBundle
from src.common.joins.feature_planning_v1.exceptions import FeatureExecutionError
from src.common.joins.feature_planning_v1.models import FeatureAction, TableFeaturePlan



class FeatureExecutor:
    def __init__(
        self,
        *,
        data_dir: Path,
        file_manager: FileManager | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._file_manager = file_manager or FileManager()

    def execute(
        self,
        *,
        bundle: InputBundle,
        table_plans: list[TableFeaturePlan],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Запускаю FeatureExecutor.execute")

        frames = self._load_frames(bundle=bundle, test_df=test_df, train_df=train_df)

        train_df = frames["train"].copy()
        test_df = frames["test"].copy()

        for table_plan in table_plans:
            logger.info(
                "Применяю TableFeaturePlan: source_table='{}', actions_count={}, requires_aggregation={}",
                table_plan.source_table,
                len(table_plan.actions),
                table_plan.requires_aggregation,
            )

            for action in table_plan.actions:
                if action.action_type == "skip":
                    logger.info(
                        "Таблица '{}' помечена как skip, пропускаю действие",
                        table_plan.source_table,
                    )
                    continue

                try:
                    train_df = self._apply_action(
                        base_df=train_df,
                        action=action,
                        frames=frames,
                    )
                except Exception:
                    logger.exception(
                        "Не удалось применить action к train_df: source_table='{}', action_type='{}'",
                        action.source_table,
                        action.action_type,
                    )

                try:
                    test_df = self._apply_action(
                        base_df=test_df,
                        action=action,
                        frames=frames,
                    )
                except Exception:
                    logger.exception(
                        "Не удалось применить action к test_df: source_table='{}', action_type='{}'",
                        action.source_table,
                        action.action_type,
                    )

        logger.info(
            "FeatureExecutor завершен: train_shape={}, test_shape={}",
            train_df.shape,
            test_df.shape,
        )
        return train_df, test_df

    def _load_frames(self, bundle: InputBundle, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        logger.info("Загружаю DataFrame для FeatureExecutor")

        tables = {
            **bundle.additional_tables,
        }

        frames: dict[str, pd.DataFrame] = {}
        for table_name, table in tables.items():
            csv_path = self._data_dir / table.file_name
            frames[table_name] = self._file_manager.read_csv(csv_path)[0]
            logger.debug(
                "Загружен DataFrame '{}': shape={}",
                table_name,
                frames[table_name].shape,
            )
        
        frames["train"] = train_df
        frames["test"] = test_df

        return frames 

    def _apply_action(
        self,
        *,
        base_df: pd.DataFrame,
        action: FeatureAction,
        frames: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        source_df = frames.get(action.source_table)
        if source_df is None:
            raise FeatureExecutionError(
                f"Не найдена source_table={action.source_table!r}"
            )

        if action.join_key not in source_df.columns:
            raise FeatureExecutionError(
                f"Ключ {action.join_key!r} отсутствует в таблице {action.source_table!r}"
            )

        if action.join_key not in base_df.columns:
            logger.warning(
                "Пропускаю action: source_table='{}', action_type='{}'. "
                "Ключ '{}' отсутствует в base_df",
                action.source_table,
                action.action_type,
                action.join_key,
            )
            return base_df

        if action.action_type == "direct_join":
            join_df = self._build_direct_join_frame(
                source_df=source_df,
                action=action,
            )
        elif action.action_type == "aggregate_numeric":
            join_df = self._build_numeric_aggregate_frame(
                source_df=source_df,
                action=action,
            )
        elif action.action_type == "aggregate_categorical":
            join_df = self._build_categorical_aggregate_frame(
                source_df=source_df,
                action=action,
            )
        else:
            raise FeatureExecutionError(
                f"Неподдерживаемый action_type={action.action_type!r}"
            )

        logger.debug(
            "Выполняю feature merge: source_table='{}', action_type='{}', base_shape={}, join_shape={}, join_key='{}'",
            action.source_table,
            action.action_type,
            base_df.shape,
            join_df.shape,
            action.join_key,
        )

        merged_df = base_df.merge(
            join_df,
            how="left",
            on=action.join_key,
            suffixes=("", "_dup"),
        )

        duplicate_columns = [column for column in merged_df.columns if column.endswith("_dup")]
        if duplicate_columns:
            merged_df = merged_df.drop(columns=duplicate_columns)

        logger.debug(
            "Feature merge завершен: source_table='{}', action_type='{}', merged_shape={}",
            action.source_table,
            action.action_type,
            merged_df.shape,
        )
        return merged_df

    def _build_direct_join_frame(
        self,
        *,
        source_df: pd.DataFrame,
        action: FeatureAction,
    ) -> pd.DataFrame:
        missing_columns = [
            column for column in action.columns if column not in source_df.columns
        ]
        if missing_columns:
            raise FeatureExecutionError(
                f"В таблице {action.source_table!r} отсутствуют колонки {missing_columns!r}"
            )

        selected_columns = [action.join_key, *action.columns]
        join_df = source_df[selected_columns].copy()

        renamed_columns: dict[str, str] = {}
        for column in action.columns:
            renamed_columns[column] = f"{action.source_table}__{column}"

        join_df = join_df.rename(columns=renamed_columns)

        logger.debug(
            "Подготовлен direct_join frame: source_table='{}', shape={}, columns={}",
            action.source_table,
            join_df.shape,
            list(join_df.columns),
        )
        return join_df

    def _build_numeric_aggregate_frame(
        self,
        *,
        source_df: pd.DataFrame,
        action: FeatureAction,
    ) -> pd.DataFrame:
        supported_aggs = {"count", "mean", "max", "min", "sum", "std", "nunique"}

        invalid_aggs = [agg for agg in action.aggregations if agg not in supported_aggs]
        if invalid_aggs:
            raise FeatureExecutionError(
                f"Недопустимые numeric aggregations: {invalid_aggs!r}"
            )

        numeric_columns: list[str] = []
        for column in action.columns:
            if column not in source_df.columns:
                continue
            if pd.api.types.is_numeric_dtype(source_df[column]):
                numeric_columns.append(column)

        if not numeric_columns:
            logger.warning(
                "Нет числовых колонок для aggregate_numeric: source_table='{}', columns={}",
                action.source_table,
                action.columns,
            )
            return source_df[[action.join_key]].drop_duplicates().copy()

        agg_spec: dict[str, list[str]] = {}
        for column in numeric_columns:
            agg_spec[column] = list(action.aggregations)

        grouped = source_df.groupby(action.join_key, dropna=False)
        aggregated = grouped.agg(agg_spec)

        aggregated.columns = [
            f"{action.source_table}__{column}__{agg_name}"
            for column, agg_name in aggregated.columns
        ]
        aggregated = aggregated.reset_index()

        logger.debug(
            "Подготовлен numeric aggregate frame: source_table='{}', shape={}, columns={}",
            action.source_table,
            aggregated.shape,
            list(aggregated.columns),
        )
        return aggregated

    def _build_categorical_aggregate_frame(
        self,
        *,
        source_df: pd.DataFrame,
        action: FeatureAction,
    ) -> pd.DataFrame:
        grouped = source_df.groupby(action.join_key, dropna=False)
        result_df = source_df[[action.join_key]].drop_duplicates().copy()

        for column in action.columns:
            if column not in source_df.columns:
                logger.warning(
                    "Колонка '{}' отсутствует в source_table='{}', пропускаю categorical action",
                    column,
                    action.source_table,
                )
                continue

            if "nunique" in action.aggregations:
                nunique_df = grouped[column].nunique(dropna=True).reset_index()
                nunique_df = nunique_df.rename(
                    columns={
                        column: f"{action.source_table}__{column}__nunique",
                    }
                )
                result_df = result_df.merge(nunique_df, how="left", on=action.join_key)

            if "most_frequent" in action.aggregations:
                mode_series = grouped[column].agg(
                    lambda series: series.mode().iloc[0]
                    if not series.mode().empty
                    else None
                )
                mode_df = mode_series.reset_index().rename(
                    columns={
                        column: f"{action.source_table}__{column}__most_frequent",
                    }
                )
                result_df = result_df.merge(mode_df, how="left", on=action.join_key)

            if "top_k_frequency" in action.aggregations:
                top_k = action.top_k or 3
                frequency_df = self._build_top_k_frequency_frame(
                    source_df=source_df,
                    join_key=action.join_key,
                    column=column,
                    prefix=f"{action.source_table}__{column}",
                    top_k=top_k,
                )
                result_df = result_df.merge(frequency_df, how="left", on=action.join_key)

        logger.debug(
            "Подготовлен categorical aggregate frame: source_table='{}', shape={}, columns={}",
            action.source_table,
            result_df.shape,
            list(result_df.columns),
        )
        return result_df

    def _build_top_k_frequency_frame(
        self,
        *,
        source_df: pd.DataFrame,
        join_key: str,
        column: str,
        prefix: str,
        top_k: int,
    ) -> pd.DataFrame:
        top_values = (
            source_df[column]
            .dropna()
            .astype(str)
            .value_counts()
            .head(top_k)
            .index
            .tolist()
        )

        result_df = source_df[[join_key]].drop_duplicates().copy()

        for value in top_values:
            indicator = (source_df[column].astype(str) == value).astype(int)
            temp_df = pd.DataFrame(
                {
                    join_key: source_df[join_key],
                    f"{prefix}__freq__{value}": indicator,
                }
            )
            temp_df = temp_df.groupby(join_key, dropna=False).sum().reset_index()
            result_df = result_df.merge(temp_df, how="left", on=join_key)

        return result_df