from __future__ import annotations

import json
import re

from loguru import logger

from src.common.io.models import InputBundle
from src.common.joins.planner_module.models import JoinPlan
from src.common.joins.feature_planning_v1.models import TableFeaturePlan
from src.common.joins.feature_planning_v1.prompt_builder import (
    TableFeaturePlanningPromptBuilder,
)
from src.common.llm_client import LLMClient
from src.common.runtime import GLOBAL_RUNTIME_BUDGET


class TableFeaturePlanningError(Exception):
    pass


class TableFeaturePlanningParser:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        prompt_builder: TableFeaturePlanningPromptBuilder | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._prompt_builder = prompt_builder or TableFeaturePlanningPromptBuilder()

    def parse(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        source_table_name: str,
    ) -> TableFeaturePlan:
        logger.info(
            "Запускаю LLM feature planning: source_table_name='{}'",
            source_table_name,
        )

        GLOBAL_RUNTIME_BUDGET.check(
            stage_name=f"table_feature_planning.before_prompt_build.{source_table_name}"
        )

        prompt = self._prompt_builder.build(
            bundle=bundle,
            join_plan=join_plan,
            source_table_name=source_table_name,
        )

        GLOBAL_RUNTIME_BUDGET.check(
            stage_name=f"table_feature_planning.before_llm_call.{source_table_name}"
        )

        raw_response = self._llm_client.invoke(
            system_prompt=(
                "Ты проектируешь baseline-признаки для табличной ML-задачи. "
                "Нужно вернуть только валидный JSON строго по заданной JSON Schema. "
                "Нельзя добавлять markdown, пояснения или комментарии."
            ),
            user_prompt=prompt,
        )

        GLOBAL_RUNTIME_BUDGET.check(
            stage_name=f"table_feature_planning.after_llm_call.{source_table_name}"
        )

        logger.debug(
            "Получен raw-ответ feature planning: source_table_name='{}', length={}",
            source_table_name,
            len(raw_response),
        )

        payload = self._extract_json_payload(raw_response)
        plan = TableFeaturePlan.model_validate_json(payload)

        logger.info(
            "Feature plan успешно распарсен: source_table_name='{}', actions_count={}",
            source_table_name,
            len(plan.actions),
        )
        return plan

    def _extract_json_payload(self, raw_response: str) -> str:
        text = raw_response.strip()
        if not text:
            raise TableFeaturePlanningError("Пустой ответ LLM при feature planning")

        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```",
            text,
            flags=re.DOTALL,
        )
        if fenced_match:
            payload = fenced_match.group(1)
            self._validate_json(payload)
            return payload

        object_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if object_match:
            payload = object_match.group(1)
            self._validate_json(payload)
            return payload

        raise TableFeaturePlanningError("Не удалось извлечь JSON из ответа LLM")

    def _validate_json(self, payload: str) -> None:
        try:
            json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.exception("LLM вернула невалидный JSON в feature planning")
            raise TableFeaturePlanningError(
                "LLM вернула невалидный JSON в feature planning"
            ) from exc