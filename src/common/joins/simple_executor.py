from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from typing import Mapping

from src.common.io.file_manager import FileManager
from src.common.io.models import InputBundle
from src.common.joins.exceptions import JoinExecutionError
from src.common.joins.planner_module.models import JoinEdge, JoinPlan


class JoinExecutor:
    def __init__(self, *, data_dir: Path, file_manager: FileManager | None = None) -> None:
        self._data_dir = data_dir
        self._file_manager = file_manager or FileManager()

    def execute(
        self,
        *,
        bundle: InputBundle,
        plan: JoinPlan,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Запускаю JoinExecutor.execute")

        frames = self._load_frames(bundle)

        train_df = frames["train"].copy()
        test_df = frames["test"].copy()

        for edge in plan.edges:
            logger.info(
                "Применяю ребро join plan: {} -> {} via {} = {}, relation_type={}, requires_aggregation={}",
                edge.parent_table,
                edge.child_table,
                edge.parent_key,
                edge.child_key,
                edge.relation_type,
                edge.requires_aggregation,
            )

            train_df = self._apply_edge(
                base_df=train_df,
                edge=edge,
                frames=frames,
                prefix=f"{edge.child_table}__",
            )
            test_df = self._apply_edge(
                base_df=test_df,
                edge=edge,
                frames=frames,
                prefix=f"{edge.child_table}__",
            )

        logger.info(
            "JoinExecutor завершен: train_shape={}, test_shape={}",
            train_df.shape,
            test_df.shape,
        )
        return train_df, test_df

    def _load_frames(self, bundle: InputBundle) -> dict[str, pd.DataFrame]:
        logger.info("Загружаю DataFrame для JoinExecutor")

        tables = {
            "train": bundle.train,
            "test": bundle.test,
            **bundle.additional_tables,
        }

        frames: Mapping[str, pd.DataFrame] = {}
        for table_name, table in tables.items():
            csv_path = self._data_dir / table.file_name
            frames[table_name] = self._file_manager.read_csv(csv_path)[0]
            logger.debug(
                "Загружен DataFrame '{}': shape={}",
                table_name,
                frames[table_name].shape,
            )

        return frames

    def _apply_edge(
        self,
        *,
        base_df: pd.DataFrame,
        edge: JoinEdge,
        frames: dict[str, pd.DataFrame],
        prefix: str,
    ) -> pd.DataFrame:
        child_df = frames.get(edge.child_table)
        if child_df is None:
            logger.warning(
                "Не найдена child-таблица '{}'. Пропускаю ребро {} -> {}",
                edge.child_table,
                edge.parent_table,
                edge.child_table,
            )
            return base_df

        resolved_parent_key = self._resolve_parent_key(base_df=base_df, edge=edge)
        if resolved_parent_key is None:
            logger.warning(
                "Пропускаю ребро {} -> {}. "
                "Не найден parent key '{}' и alias '{}__{}'.",
                edge.parent_table,
                edge.child_table,
                edge.parent_key,
                edge.parent_table,
                edge.parent_key,
            )
            return base_df

        if edge.requires_aggregation:
            join_df = self._aggregate_child_table(
                child_df=child_df,
                child_key=edge.child_key,
                prefix=prefix,
            )
        else:
            join_df = self._prepare_non_aggregated_child(
                child_df=child_df,
                child_key=edge.child_key,
                prefix=prefix,
            )

        logger.debug(
            "Выполняю merge: base_shape={}, join_shape={}, left_on='{}', right_on='{}'",
            base_df.shape,
            join_df.shape,
            resolved_parent_key,
            edge.child_key,
        )

        try:
            merged_df = base_df.merge(
                join_df,
                how="left",
                left_on=resolved_parent_key,
                right_on=edge.child_key,
                suffixes=("", "_dup"),
            )
        except Exception:
            logger.exception(
                "Ошибка merge для ребра {} -> {} via {} = {}",
                edge.parent_table,
                edge.child_table,
                resolved_parent_key,
                edge.child_key,
            )
            return base_df

        duplicate_columns = [column for column in merged_df.columns if column.endswith("_dup")]
        if duplicate_columns:
            merged_df = merged_df.drop(columns=duplicate_columns)

        logger.debug("Merge завершен: merged_shape={}", merged_df.shape)
        return merged_df
    
    def _resolve_parent_key(self, *, base_df: pd.DataFrame, edge: JoinEdge) -> str | None:
        if edge.parent_key in base_df.columns:
            return edge.parent_key

        prefixed_key = f"{edge.parent_table}__{edge.parent_key}"
        if prefixed_key in base_df.columns:
            return prefixed_key

        return None

    def _prepare_non_aggregated_child(
        self,
        *,
        child_df: pd.DataFrame,
        child_key: str,
        prefix: str,
    ) -> pd.DataFrame:
        prepared = child_df.copy()

        renamed_columns: dict[str, str] = {}
        for column in prepared.columns:
            if column == child_key:
                continue
            renamed_columns[column] = f"{prefix}{column}"

        prepared = prepared.rename(columns=renamed_columns)

        logger.debug(
            "Подготовлена неагрегированная child-таблица: child_key='{}', shape={}",
            child_key,
            prepared.shape,
        )
        return prepared

    def _aggregate_child_table(
        self,
        *,
        child_df: pd.DataFrame,
        child_key: str,
        prefix: str,
    ) -> pd.DataFrame:
        if child_key not in child_df.columns:
            raise JoinExecutionError(
                f"Ключ {child_key!r} отсутствует в child-таблице"
            )

        work_df = child_df.copy()

        numeric_columns = [
            column
            for column in work_df.columns
            if column != child_key and pd.api.types.is_numeric_dtype(work_df[column])
        ]
        categorical_columns = [
            column
            for column in work_df.columns
            if column != child_key and not pd.api.types.is_numeric_dtype(work_df[column])
        ]

        agg_spec: dict[str, list[str] | str] = {}

        for column in numeric_columns:
            agg_spec[column] = ["mean", "max", "min", "nunique"]

        for column in categorical_columns:
            agg_spec[column] = ["nunique"]

        grouped = work_df.groupby(child_key, dropna=False)

        if agg_spec:
            aggregated = grouped.agg(agg_spec)
            aggregated.columns = [
                f"{prefix}{column}__{agg_name}"
                for column, agg_name in aggregated.columns # .to_flat_index()
            ]
            aggregated = aggregated.reset_index()
        else:
            aggregated = grouped.size().reset_index(name=f"{prefix}row_count")

        row_count_df = grouped.size().reset_index(name=f"{prefix}row_count")

        if child_key in aggregated.columns:
            result = aggregated.merge(row_count_df, how="left", on=child_key)
        else:
            result = row_count_df

        logger.debug(
            "Агрегирована child-таблица: child_key='{}', input_shape={}, output_shape={}, numeric_columns={}, categorical_columns={}",
            child_key,
            child_df.shape,
            result.shape,
            numeric_columns,
            categorical_columns,
        )
        return result