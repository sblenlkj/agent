from __future__ import annotations

import asyncio

from loguru import logger

from src.common.feature_codegen.models import GeneratedFeatureCode, GeneratedFeatureCodeResponse
from src.common.feature_codegen.prompt_builder import FeatureCodePromptBuilder
from src.common.feature_ideas_generation.models import FeatureIdea
from src.common.io.models import InputBundle
from src.common.joins.planner_module.models import JoinPlan
from src.common.llm_client import LLMClient


class FeatureCodeGenerationService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._prompt_builder = FeatureCodePromptBuilder()

    def generate_one(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        feature_idea: FeatureIdea,
    ) -> GeneratedFeatureCode:
        logger.info(
            "Генерация pandas-кода начата: feature_idea={}",
            feature_idea.title,
        )

        user_prompt = self._prompt_builder.build(
            bundle=bundle,
            join_plan=join_plan,
            feature_idea=feature_idea,
        )

        response = self._llm_client.invoke_json(
            user_prompt=user_prompt,
            response_model=GeneratedFeatureCodeResponse,
        )

        logger.info(
            "Генерация pandas-кода завершена: title={}, merge_back_keys={}",
            response.generated_feature_code.title,
            response.generated_feature_code.merge_back_keys,
        )
        return response.generated_feature_code

    async def agenerate_one(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        feature_idea: FeatureIdea,
    ) -> GeneratedFeatureCode:
        logger.info(
            "Async генерация pandas-кода начата: feature_idea={}",
            feature_idea.title,
        )

        user_prompt = self._prompt_builder.build(
            bundle=bundle,
            join_plan=join_plan,
            feature_idea=feature_idea,
        )

        response = await self._llm_client.ainvoke_json(
            user_prompt=user_prompt,
            response_model=GeneratedFeatureCodeResponse,
        )

        logger.info(
            "Async генерация pandas-кода завершена: title={}, merge_back_keys={}",
            response.generated_feature_code.title,
            response.generated_feature_code.merge_back_keys,
        )
        return response.generated_feature_code

    async def agenerate_many(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        feature_ideas: list[FeatureIdea],
    ) -> list[GeneratedFeatureCode]:
        logger.info(
            "Async генерация нескольких pandas-кодов начата: feature_ideas_count={}",
            len(feature_ideas),
        )

        tasks = [
            self.agenerate_one(
                bundle=bundle,
                join_plan=join_plan,
                feature_idea=feature_idea,
            )
            for feature_idea in feature_ideas
        ]

        results = await asyncio.gather(*tasks)
        logger.info(
            "Async генерация нескольких pandas-кодов завершена: code_count={}",
            len(results),
        )
        return results