from __future__ import annotations

import json

from src.common.archive.compilation.parser import CompiledFeatureSpecsResponse
from src.common.feature_ideas_generation.models import FeatureIdea
from src.common.io.models import InputBundle
from src.common.joins.planner_module.models import JoinPlan


class FeatureIdeaCompilationPromptBuilder:
    def build(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        feature_idea: FeatureIdea,
    ) -> str:
        schema_json = json.dumps(
            CompiledFeatureSpecsResponse.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        table_names: list[str] = [
            bundle.train.name,
            bundle.test.name,
            *bundle.additional_tables.keys(),
        ]

        edge_blocks: list[str] = []
        for edge in join_plan.edges:
            edge_blocks.append(edge.to_prompt)

        feature_idea_json = json.dumps(
            feature_idea.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )

        example_json = {
            "compiled_feature_specs": [
                {
                    "title": "Агрегации истории пользователя по категориям текущего товара",
                    "hypothesis": (
                        "Если пользователь часто покупал товары из той же категории, "
                        "вероятность покупки текущего товара выше"
                    ),
                    "required_tables": [
                        "orders",
                        "order_items",
                        "products",
                        "aisles",
                        "departments",
                    ],
                    "join_paths": [
                        ["train", "orders"],
                        ["train", "orders", "order_items"],
                        ["train", "orders", "order_items", "products"],
                    ],
                    "aggregations": [
                        {
                            "source_table": "order_items",
                            "source_column": "product_id",
                            "operation": "count",
                            "group_by_columns": ["user_id", "department_id"],
                            "feature_name": "user_department_purchase_count",
                            "filter_expression": None,
                            "notes": (
                                "Нужно протянуть department_id из products к истории покупок"
                            ),
                        },
                        {
                            "source_table": "order_items",
                            "source_column": "product_id",
                            "operation": "nunique",
                            "group_by_columns": ["user_id", "aisle_id"],
                            "feature_name": "user_aisle_distinct_products",
                            "filter_expression": None,
                            "notes": (
                                "Нужно протянуть aisle_id из products к истории покупок"
                            ),
                        },
                    ],
                    "notes": (
                        "Спецификация опирается на историю покупок пользователя "
                        "и категориальные атрибуты текущего товара"
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
Ты превращаешь одну идею признаков в строгую исполняемую JSON-спецификацию.

Контекст:
- задача бинарной классификации на табличных данных
- join planner уже построил допустимые связи между таблицами
- твоя задача не придумать новые идеи, а перевести одну идею в исполнимый набор агрегаций
- не пиши код
- не предлагай несуществующие таблицы или колонки
- используй только доступные пути джойнов
- если идея слишком широкая, разбей ее на несколько компактных compiled_feature_specs
- ответ должен строго соответствовать JSON Schema ниже

Доступные таблицы:
{json.dumps(table_names, ensure_ascii=False, indent=2)}

Доступные связи:
{chr(10).join(edge_blocks)}

Идея для компиляции:
{feature_idea_json}

Допустимые операции агрегации:
- count
- nunique
- sum
- mean
- min
- max
- std
- ratio
- mode_ratio

Пример хорошего ответа:
{example_json_str}

JSON Schema ответа:
{schema_json}
""".strip()