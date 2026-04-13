from __future__ import annotations

import json

from loguru import logger

from src.common.io.models import InputBundle
from src.common.readme.models import ReadmeParseResponse


class ReadmePromptBuilder:
    def build(self, bundle: InputBundle) -> str:
        logger.debug("Собираю prompt для парсинга README")

        schema_json = json.dumps(
            ReadmeParseResponse.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        lines: list[str] = [
            "Ниже дан README и список уже известных таблиц и колонок.",
            "Нужно извлечь только семантические описания и вернуть строго валидный JSON.",
            "",
            "Правила ответа:",
            "1. Верни только JSON без markdown, без пояснений и без комментариев.",
            "2. Используй только table_name и column_name из списка ниже.",
            "3. Не придумывай новые таблицы и новые колонки.",
            "4. Если описание отсутствует или не следует явно из README, верни null.",
            "5. allowed_values заполняй только если допустимые значения явно указаны в README.",
            "6. Описания должны быть короткими, точными и полезными для понимания схемы данных.",
            "",
            "Известные таблицы и колонки:",
        ]

        for table_name, column_names in self._collect_table_columns(bundle).items():
            lines.append(f"- {table_name}: {', '.join(column_names)}")

        lines.extend(
            [
                "",
                "JSON Schema ответа:",
                schema_json,
                "",
                "README:",
                bundle.readme_text,
            ]
        )

        prompt = "\n".join(lines)
        logger.debug("Prompt для README собран, длина={} символов", len(prompt))
        return prompt

    def _collect_table_columns(self, bundle: InputBundle) -> dict[str, list[str]]:
        tables = {
            bundle.train.name: [column.name for column in bundle.train.columns],
            bundle.test.name: [column.name for column in bundle.test.columns],
        }

        for table_name, table in bundle.additional_tables.items():
            tables[table_name] = [column.name for column in table.columns]

        return tables