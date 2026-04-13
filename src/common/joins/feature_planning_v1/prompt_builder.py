from __future__ import annotations

import json

from loguru import logger

from src.common.io.models import InputBundle, SourceTable
from src.common.joins.planner_module.models import JoinEdge, JoinPlan
from src.common.joins.feature_planning_v1.models import TableFeaturePlan


class TableFeaturePlanningPromptBuilder:
    def build(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        source_table_name: str,
    ) -> str:
        logger.debug(
            "Собираю prompt для feature planning: source_table_name='{}'",
            source_table_name,
        )

        source_table = self._get_table(bundle=bundle, table_name=source_table_name)
        join_edge = self._get_join_edge(join_plan=join_plan, source_table_name=source_table_name)

        schema_json = json.dumps(
            TableFeaturePlan.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        train_columns = self._render_columns(bundle.train)
        source_columns = self._render_columns(source_table)

        lines: list[str] = [
            "Ниже дана одна таблица из train-centric join plan.",
            "Нужно предложить безопасный baseline-план извлечения признаков из этой таблицы для задачи бинарной классификации.",
            "",
            "Важные ограничения:",
            "1. Верни строго JSON без markdown и без пояснений.",
            "2. Не придумывай новые колонки.",
            "3. Используй только колонки, реально существующие в source_table.",
            "4. Выбирай только простые baseline-действия.",
            "5. Не предлагай target leakage.",
            "6. Не предлагай embeddings, текстовые модели, сложные графовые признаки и внешние данные.",
            "7. Если таблица почти бесполезна для baseline, можно вернуть action_type=skip.",
            "8. Обычно выбирай не более 3 actions.",
            "",
            "Допустимые action_type:",
            "- direct_join: напрямую присоединить полезные поля без агрегации",
            "- aggregate_numeric: агрегировать числовые поля по join_key",
            "- aggregate_categorical: агрегировать категориальные поля по join_key",
            "- skip: пропустить таблицу",
            "",
            "Допустимые агрегации для aggregate_numeric:",
            "- count",
            "- mean",
            "- max",
            "- min",
            "- sum",
            "- std",
            "- nunique",
            "",
            "Допустимые агрегации для aggregate_categorical:",
            "- nunique",
            "- top_k_frequency",
            "- top_k_one_hot",
            "- most_frequent",
            "",
            "Информация о задаче:",
            f"- task_description: {bundle.readme_text[:1500]}",
            "",
            "Информация о train:",
            f"- train_table: {bundle.train.name}",
            f"- train_description: {bundle.train.description!r}",
            f"- train_columns: {train_columns}",
            "",
            "Информация о source_table:",
            f"- source_table: {source_table.name}",
            f"- source_description: {source_table.description!r}",
            f"- source_columns: {source_columns}",
            "",
            "Информация о связи с train:",
            f"- parent_table: {join_edge.parent_table}",
            f"- child_table: {join_edge.child_table}",
            f"- parent_key: {join_edge.parent_key}",
            f"- child_key: {join_edge.child_key}",
            f"- relation_type: {join_edge.relation_type}",
            f"- requires_aggregation: {join_edge.requires_aggregation}",
            f"- join_path: {join_edge.path_from_train}",
            f"- reason: {join_edge.reason}",
            "",
            "Смысл вопроса:",
            "Нужно понять, какие именно признаки стоит построить из этой таблицы для baseline-модели CatBoost.",
            "Если таблица уже агрегированная и достаточно clean, чаще подходит direct_join.",
            "Если это история событий или many-to-many / one-to-many связь, обычно нужна агрегация.",
            "",
            "JSON Schema ответа:",
            schema_json,
        ]

        prompt = "\n".join(lines)
        logger.debug(
            "Prompt для feature planning собран: source_table_name='{}', length={}",
            source_table_name,
            len(prompt),
        )
        return prompt

    def _get_table(self, *, bundle: InputBundle, table_name: str) -> SourceTable:
        if table_name == bundle.train.name:
            return bundle.train
        if table_name == bundle.test.name:
            return bundle.test
        if table_name in bundle.additional_tables:
            return bundle.additional_tables[table_name]
        raise ValueError(f"Таблица {table_name!r} не найдена в InputBundle")

    def _get_join_edge(self, *, join_plan: JoinPlan, source_table_name: str) -> JoinEdge:
        for edge in join_plan.edges:
            if edge.child_table == source_table_name:
                return edge
        raise ValueError(f"Для таблицы {source_table_name!r} не найдено ребро в JoinPlan")

    def _render_columns(self, table: SourceTable) -> list[dict[str, object]]:
        rendered: list[dict[str, object]] = []
        for column in table.columns:
            rendered.append(
                {
                    "name": column.name,
                    "dtype": column.dtype,
                    "description": column.description,
                    "non_null_ratio": column.non_null_ratio,
                    "unique_ratio": column.unique_ratio,
                    "allowed_values": getattr(column, "allowed_values", None),
                }
            )
        return rendered