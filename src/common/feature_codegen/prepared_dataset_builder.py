from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.common.feature_codegen.executor import FeatureCodeExecutor
from src.common.feature_codegen.models import GeneratedFeatureCode
from src.common.io.file_manager import FileManager

class TableFrameRepository:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._cache: dict[str, pd.DataFrame] = {}

    def get_table(self, table_name: str) -> pd.DataFrame:
        csv_path = self._data_dir / f"{table_name}.csv"

        if table_name in self._cache:
            return self._cache[table_name].copy()

        if not csv_path.exists():
            raise FileNotFoundError(f"Таблица не найдена: {csv_path}")

        df = FileManager.read_csv(Path(csv_path))[0]
        self._cache[table_name] = df
        logger.info("Таблица загружена: {} rows={}, cols={}", table_name, len(df), len(df.columns))
        return df.copy()

    def get_all_tables(self) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        for csv_path in sorted(self._data_dir.glob("*.csv")):
            result[csv_path.stem] = self.get_table(csv_path.stem)
        return result


class GeneratedFeaturesDatasetBuilder:
    def __init__(
        self,
        *,
        data_dir: Path,
    ) -> None:
        self._repository = TableFrameRepository(data_dir=data_dir)
        self._executor = FeatureCodeExecutor()

        self._train_df = self._repository.get_table("train")
        self._test_df = self._repository.get_table("test")
        self._tables = self._repository.get_all_tables()

    @property
    def train_df(self) -> pd.DataFrame:
        return self._train_df

    @property
    def test_df(self) -> pd.DataFrame:
        return self._test_df

    def apply_generated_code(
        self,
        *,
        generated_feature_code: GeneratedFeatureCode,
    ) -> None:
        feature_df = self._executor.execute(
            generated_feature_code=generated_feature_code,
            train_df=self._train_df,
            tables=self._tables,
        )

        self._train_df = self._safe_merge(
            base_df=self._train_df,
            feature_df=feature_df,
            merge_back_keys=generated_feature_code.merge_back_keys,
            title=generated_feature_code.title,
            dataset_name="train",
        )
        self._test_df = self._safe_merge(
            base_df=self._test_df,
            feature_df=feature_df,
            merge_back_keys=generated_feature_code.merge_back_keys,
            title=generated_feature_code.title,
            dataset_name="test",
        )

    def apply_many(
        self,
        *,
        generated_feature_codes: list[GeneratedFeatureCode],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        for generated_feature_code in generated_feature_codes:
            self.apply_generated_code(
                generated_feature_code=generated_feature_code,
            )

        return self._train_df, self._test_df

    def _safe_merge(
        self,
        *,
        base_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        merge_back_keys: list[str],
        title: str,
        dataset_name: str,
    ) -> pd.DataFrame:
        missing_base_keys = [
            key for key in merge_back_keys
            if key not in base_df.columns
        ]
        if missing_base_keys:
            raise ValueError(
                f"В {dataset_name} отсутствуют merge keys {missing_base_keys} "
                f"для generated feature {title!r}"
            )

        missing_feature_keys = [
            key for key in merge_back_keys
            if key not in feature_df.columns
        ]
        if missing_feature_keys:
            raise ValueError(
                f"В feature_df отсутствуют merge keys {missing_feature_keys} "
                f"для generated feature {title!r}"
            )

        before_rows = len(base_df)
        merged_df = base_df.merge(
            feature_df,
            on=merge_back_keys,
            how="left",
        )

        after_rows = len(merged_df)
        if before_rows != after_rows:
            raise ValueError(
                f"После merge generated feature {title!r} число строк изменилось: "
                f"{before_rows} -> {after_rows}"
            )

        logger.info(
            "Feature merge завершён: dataset={}, title={}, rows={}, cols={}",
            dataset_name,
            title,
            len(merged_df),
            len(merged_df.columns),
        )
        return merged_df