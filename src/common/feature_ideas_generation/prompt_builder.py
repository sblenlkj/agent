from __future__ import annotations

import json

from src.common.feature_ideas_generation.parser import FeatureIdeasResponse
from src.common.io.models import ColumnState, InputBundle
from src.common.joins.planner_module.models import JoinPlan


class FeatureIdeasPromptBuilder:
    def build(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        max_feature_ideas: int,
    ) -> str:
        schema_json = json.dumps(
            FeatureIdeasResponse.model_json_schema(),
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
            edge_blocks.append(edge.to_prompt)

        example_json = {
            "feature_ideas": [
                {
                    "title": "История покупок пользователя по категориям текущего товара",
                    "hypothesis": (
                        "Пользователь с большей вероятностью купит товар, если раньше "
                        "часто покупал товары из той же категории или отдела"
                    ),
                    "priority": 1,
                    "required_tables": [
                        "orders",
                        "order_items",
                        "products",
                        "aisles",
                        "departments",
                    ],
                    "required_join_paths": [
                        ["train", "orders"],
                        ["train", "orders", "order_items"],
                        ["train", "orders", "order_items", "products"],
                    ],
                    "candidate_feature_families": [
                        "category_purchase_count",
                        "category_purchase_share",
                        "category_distinct_products",
                        "category_repeat_behavior",
                    ],
                    "notes": (
                        "Нужно связать историю заказов пользователя с категорией "
                        "или отделом текущего товара из train"
                    ),
                }
            ]
        }

        example_json_str = json.dumps(
            example_json,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
Ты помогаешь спроектировать идеи признаков для задачи бинарной классификации на табличных данных.

Контекст:
- есть train и test
- есть дополнительные таблицы
- уже построен deterministic join plan
- target находится в train
- нужно избегать leakage

Нужно:
1. Предложить не более {max_feature_ideas} полезных идей признаков. Одна идея это одна аггрегация таблиц. Т е найти соединить две таблицы и найти среднее, мин/макс, какого-то признака это считается за 1.
2. Каждая идея должна быть реализуема через доступные таблицы и пути джойнов.
3. Идеи должны быть практичными для табличного ML.
4. Не пиши код.
5. Верни только JSON.
6. Ответ должен строго соответствовать JSON Schema ниже.

Описание таблиц:
{chr(10).join(table_blocks)}

Доступные связи:
{chr(10).join(edge_blocks)}

Пример хорошей идеи:
{example_json_str}

JSON Schema ответа:
{schema_json}
""".strip()

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