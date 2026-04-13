from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.common.io.exceptions import MissingRequiredFileError, RepositoryError
from src.common.io.file_manager import FileManager
from src.common.io.models import InputBundle, SourceTable
from src.common.schema.models import ColumnState


class InputRepository:
    def __init__(self, data_path: Path):
        self.data_path = data_path

    def load(self) -> InputBundle:
        logger.info("Loading input bundle from '{}' folder", self.data_path)

        readme_path = self.data_path / "readme.txt"
        if not readme_path.exists():
            logger.error("README file not found: {}", readme_path)
            raise MissingRequiredFileError(f"README file not found: {readme_path}")

        try:
            readme_text = FileManager.read_text(readme_path)
            tables = self._load_all_tables()

            if "train" not in tables:
                logger.error("train.csv was not found in {}", self.data_path)
                raise MissingRequiredFileError("train.csv was not found in data directory")

            if "test" not in tables:
                logger.error("test.csv was not found in {}", self.data_path)
                raise MissingRequiredFileError("test.csv was not found in data directory")

            train = tables.pop("train")
            test = tables.pop("test")

            bundle = InputBundle(
                readme_text=readme_text,
                train=train,
                test=test,
                additional_tables=tables,
            )

            logger.info(
                "Loaded input bundle: train_rows={}, test_rows={}, additional_tables={}",
                bundle.train.row_count,
                bundle.test.row_count,
                list(bundle.additional_tables.keys()),
            )
            return bundle
        except Exception as exc:
            logger.exception("Failed to build input bundle from {}", self.data_path)
            raise RepositoryError("Failed to build input bundle") from exc

    def _load_all_tables(self) -> dict[str, SourceTable]:
        result: dict[str, SourceTable] = {}

        for file_path in FileManager.list_csv_files(self.data_path):
            table_name = file_path.stem
            logger.info("Loading table '{}' from {}", table_name, file_path)

            df, separator = FileManager.read_csv(file_path)
            result[table_name] = self._build_source_table(
                name=table_name,
                df=df,
                file_name=file_path.name,
                separator=separator,
            )

        logger.debug("Loaded tables: {}", list(result.keys()))
        return result
        

    def _build_source_table(
        self,
        *,
        name: str,
        df: pd.DataFrame,
        file_name: str,
        separator: str,
    ) -> SourceTable:
        columns = self._build_column_states(df)
        candidate_keys = self._detect_candidate_keys(columns)

        table = SourceTable(
            name=name,
            file_name=file_name,
            separator=separator,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            candidate_keys=candidate_keys,
        )

        logger.info(
            "Built SourceTable '{}': rows={}, cols={}, candidate_keys={}",
            name,
            table.row_count,
            table.column_count,
            table.candidate_keys,
        )
        return table

    def _build_column_states(self, df: pd.DataFrame) -> list[ColumnState]:
        row_count = len(df)
        safe_row_count = max(row_count, 1)

        result: list[ColumnState] = []
        for col_name in df.columns:
            series = df[col_name]
            result.append(
                ColumnState(
                    name=col_name,
                    dtype=str(series.dtype),
                    non_null_ratio=float(series.notna().mean()),
                    unique_ratio=float(series.nunique(dropna=True) / safe_row_count),
                )
            )

        logger.debug("Built {} column states", len(result))
        return result

    def _detect_candidate_keys(self, columns: list[ColumnState]) -> list[str]:
        candidate_keys = [
            column.name
            for column in columns
            if column.name == "id" or column.name.endswith("_id")
        ]
        logger.debug("Detected candidate keys: {}", candidate_keys)
        return candidate_keys
