from __future__ import annotations

from collections.abc import Mapping

from loguru import logger

from src.common.io.models import InputBundle, SourceTable
from src.common.readme.models import ReadmeParseResponse, ReadmeTableDescription
from src.common.schema.models import ColumnState


class ReadmeMergeError(Exception):
    pass


class ReadmeBundleMerger:
    def merge(
        self,
        *,
        bundle: InputBundle,
        response: ReadmeParseResponse,
    ) -> InputBundle:
        logger.info("Начинаю merge результата парсинга README в InputBundle")

        table_updates: dict[str, ReadmeTableDescription] = {
            table_description.table_name: table_description
            for table_description in response.tables
        }

        updated_train = self._merge_table(
            table=bundle.train,
            table_updates=table_updates,
        )
        updated_test = self._merge_table(
            table=bundle.test,
            table_updates=table_updates,
        )

        updated_additional_tables: dict[str, SourceTable] = {}
        for table_name, table in bundle.additional_tables.items():
            updated_additional_tables[table_name] = self._merge_table(
                table=table,
                table_updates=table_updates,
            )

        merged_bundle = bundle.model_copy(
            update={
                "train": updated_train,
                "test": updated_test,
                "additional_tables": updated_additional_tables,
            }
        )

        logger.info("Merge результата README завершен")
        return merged_bundle


    def _merge_table(
        self,
        *,
        table: SourceTable,
        table_updates: Mapping[str, ReadmeTableDescription],
    ) -> SourceTable:
        table_update = table_updates.get(table.name)
        if table_update is None:
            logger.debug(
                "Для таблицы '{}' описание не пришло. table_columns={}",
                table.name,
                [column.name for column in table.columns],
            )
            return table

        column_updates = {
            column_update.column_name: column_update
            for column_update in table_update.columns
            if column_update.table_name == table.name
        }

        updated_columns: list[ColumnState] = []
        updated_column_names: list[str] = []
        missing_column_names: list[str] = []

        for column in table.columns:
            column_update = column_updates.get(column.name)
            if column_update is None:
                updated_columns.append(column)
                missing_column_names.append(column.name)
                continue

            update_payload: dict[str, object] = {
                "description": column_update.description,
            }

            if hasattr(column, "allowed_values"):
                update_payload["allowed_values"] = column_update.allowed_values

            updated_columns.append(column.model_copy(update=update_payload))
            updated_column_names.append(column.name)

        updated_table = table.model_copy(
            update={
                "description": table_update.description,
                "columns": updated_columns,
            }
        )

        logger.debug(
            "Таблица '{}' обновлена: table_description={}, updated_columns={}/{}, updated_column_names={}, missing_column_names={}",
            table.name,
            bool(table_update.description),
            len(updated_column_names),
            len(table.columns),
            updated_column_names,
            missing_column_names,
        )
        return updated_table