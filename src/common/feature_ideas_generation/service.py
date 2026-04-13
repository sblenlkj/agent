from __future__ import annotations

from loguru import logger

from src.common.feature_ideas_generation.parser import FeatureIdeasResponse
from src.common.feature_ideas_generation.prompt_builder import FeatureIdeasPromptBuilder
from src.common.feature_ideas_generation.models import FeatureIdea
from src.common.io.models import InputBundle
from src.common.joins.planner_module.models import JoinPlan
from src.common.llm_client import LLMClient


class FeatureIdeasGenerationService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._prompt_builder = FeatureIdeasPromptBuilder()

    def generate(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        max_feature_ideas: int = 6,
    ) -> list[FeatureIdea]:
        logger.info(
            "Генерация feature ideas начата: max_feature_ideas={}",
            max_feature_ideas,
        )

        user_prompt = self._prompt_builder.build(
            bundle=bundle,
            join_plan=join_plan,
            max_feature_ideas=max_feature_ideas,
        )

        response = self._llm_client.invoke_json(
            user_prompt=user_prompt,
            response_model=FeatureIdeasResponse,
        )

        logger.info(
            "Генерация feature ideas завершена: count={}",
            len(response.feature_ideas),
        )
        return response.feature_ideas

    async def agenerate(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        max_feature_ideas: int = 6,
    ) -> list[FeatureIdea]:
        logger.info(
            "Async генерация feature ideas начата: max_feature_ideas={}",
            max_feature_ideas,
        )

        user_prompt = self._prompt_builder.build(
            bundle=bundle,
            join_plan=join_plan,
            max_feature_ideas=max_feature_ideas,
        )

        response = await self._llm_client.ainvoke_json(
            user_prompt=user_prompt,
            response_model=FeatureIdeasResponse,
        )

        logger.info(
            "Async генерация feature ideas завершена: count={}",
            len(response.feature_ideas),
        )
        return response.feature_ideas