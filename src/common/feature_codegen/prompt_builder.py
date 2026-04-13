from __future__ import annotations

import json

from src.common.feature_codegen.models import GeneratedFeatureCodeResponse
from src.common.feature_ideas_generation.models import FeatureIdea
from src.common.io.models import ColumnState, InputBundle
from src.common.joins.planner_module.models import JoinPlan


class FeatureCodePromptBuilder:
    def build(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        feature_idea: FeatureIdea,
    ) -> str:
        schema_json = json.dumps(
            GeneratedFeatureCodeResponse.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        table_blocks: list[str] = [
            self._build_table_block(
                table_name=bundle.train.name,
                table_description=bundle.train.description,
                columns=bundle.train.columns,
            ),
            self._build_table_block(
                table_name=bundle.test.name,
                table_description=bundle.test.description,
                columns=bundle.test.columns,
            ),
        ]

        for table_name, table in bundle.additional_tables.items():
            table_blocks.append(
                self._build_table_block(
                    table_name=table_name,
                    table_description=table.description,
                    columns=table.columns,
                )
            )

        edge_blocks: list[str] = []
        for edge in join_plan.edges:
            edge_blocks.append(
                edge.to_prompt
            )

        feature_idea_json = json.dumps(
            feature_idea.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )

        example_response = {
            "generated_feature_code": {
                "title": "Частота повторных покупок продукта пользователем",
                "merge_back_keys": ["user_id", "product_id"],
                "code": (
                    "def build_feature(train_df, tables):\n"
                    "    order_items_df = tables['order_items'].copy()\n"
                    "    orders_df = tables['orders'][['order_id', 'user_id']].copy()\n"
                    "    history_df = order_items_df.merge(\n"
                    "        orders_df,\n"
                    "        on='order_id',\n"
                    "        how='left',\n"
                    "    )\n"
                    "    feature_df = (\n"
                    "        history_df.groupby(['user_id', 'product_id'], dropna=False)['reordered']\n"
                    "        .agg(\n"
                    "            user_product_order_count='size',\n"
                    "            user_product_reorder_rate=lambda x: (x == 1).mean(),\n"
                    "        )\n"
                    "        .reset_index()\n"
                    "    )\n"
                    "    return feature_df\n"
                ),
                "notes": (
                    "Нужно протянуть user_id из orders в историю order_items по order_id"
                ),
            }
        }

        example_response_json = json.dumps(
            example_response,
            ensure_ascii=False,
            indent=2,
        )

        prompt = f"""
Ты генерируешь короткий и исполнимый pandas-код для построения одного feature dataframe.

Контекст:
- есть train_df
- есть словарь tables, где ключ — имя таблицы, значение — pandas DataFrame
- уже известны доступные таблицы и связи между ними
- нужно сгенерировать только функцию build_feature(train_df, tables)
- функция должна вернуть уже агрегированный DataFrame с уникальными строками по merge_back_keys
- не нужно писать код для merge в train/test
- не нужно читать файлы с диска
- не нужно обучать модель
- не нужно печатать лог или использовать print
- не нужно изменять входные DataFrame inplace
- не нужно писать explanation вне JSON

Жёсткие правила:
1. Верни только JSON.
2. Ответ должен строго соответствовать JSON Schema ниже.
3. В поле code должна быть определена функция:
   def build_feature(train_df, tables) -> pd.DataFrame
4. Используй только pandas и numpy.
5. Не импортируй os, sys, pathlib, subprocess, builtins, pickle и другие системные библиотеки.
6. Не используй eval, exec, open, __import__, globals, locals.
7. Не записывай файлы.
8. Возвращаемый dataframe должен содержать все merge_back_keys.
9. Код должен быть коротким и практичным.
10. Перед groupby убедись, что все нужные колонки реально присутствуют в dataframe.
11. Нельзя использовать колонку, если её нет в конкретной таблице.
12. Если нужной колонки нет в исходной таблице, сначала сделай merge с таблицей, где она есть.
13. Если идея слишком широкая, реализуй только её простую и полезную часть.
14. Используй required_tables и required_join_paths как ориентир, но при необходимости аккуратно дострой один очевидный промежуточный merge по реальным ключам.

Критически важные подсказки по этому датасету:
- order_items не содержит user_id
- user_id для истории покупок можно получить через merge order_items -> orders по order_id
- products содержит product_id, aisle_id, department_id
- aisle_id и department_id обычно нужно получать через merge с products по product_id
- orders содержит user_id, order_id, days_since_prior_order, order_dow, order_hour_of_day
- users уже содержит агрегаты на уровне user_id и обычно не требует дополнительного history merge

Доступные таблицы и их колонки:
{chr(10).join(table_blocks)}

Доступные связи:
{chr(10).join(edge_blocks)}

FeatureIdea:
{feature_idea_json}

Пример хорошего ответа:
{example_response_json}

JSON Schema ответа:
{schema_json}
""".strip()

        return prompt

    def _build_table_block(
        self,
        *,
        table_name: str,
        table_description: str | None,
        columns: list[ColumnState],
    ) -> str:
        column_lines: list[str] = []

        for column in columns:
            allowed_values_part = (
                f", allowed_values={column.allowed_values!r}"
                if column.allowed_values
                else ""
            )

            column_lines.append(
                (
                    f"  - {column.name}: "
                    f"dtype={column.dtype!r}, "
                    f"description={column.description!r}, "
                    f"non_null_ratio={column.non_null_ratio}, "
                    f"unique_ratio={column.unique_ratio}"
                    f"{allowed_values_part}"
                )
            )

        return (
            f"Таблица: {table_name}\n"
            f"Описание: {table_description!r}\n"
            f"Колонки:\n"
            f"{chr(10).join(column_lines)}"
        )