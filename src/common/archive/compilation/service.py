from __future__ import annotations

import asyncio

from loguru import logger

from src.common.archive.compilation.parser import CompiledFeatureSpecsResponse
from src.common.archive.compilation.prompt_builder import (
    FeatureIdeaCompilationPromptBuilder,
)
from src.common.archive.compilation.models import CompiledFeatureSpec
from src.common.feature_ideas_generation.models import FeatureIdea
from src.common.io.models import InputBundle
from src.common.joins.planner_module.models import JoinPlan
from src.common.llm_client import LLMClient


class FeatureIdeaCompilationService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._prompt_builder = FeatureIdeaCompilationPromptBuilder()

    def compile_one(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        feature_idea: FeatureIdea,
    ) -> list[CompiledFeatureSpec]:
        logger.info(
            "Компиляция feature idea начата: title={}",
            feature_idea.title,
        )

        user_prompt = self._prompt_builder.build(
            bundle=bundle,
            join_plan=join_plan,
            feature_idea=feature_idea,
        )

        response = self._llm_client.invoke_json(
            user_prompt=user_prompt,
            response_model=CompiledFeatureSpecsResponse,
        )

        logger.info(
            "Компиляция feature idea завершена: title={}, specs_count={}",
            feature_idea.title,
            len(response.compiled_feature_specs),
        )
        return response.compiled_feature_specs

    async def acompile_one(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        feature_idea: FeatureIdea,
    ) -> list[CompiledFeatureSpec]:
        logger.info(
            "Async компиляция feature idea начата: title={}",
            feature_idea.title,
        )

        user_prompt = self._prompt_builder.build(
            bundle=bundle,
            join_plan=join_plan,
            feature_idea=feature_idea,
        )

        response = await self._llm_client.ainvoke_json(
            user_prompt=user_prompt,
            response_model=CompiledFeatureSpecsResponse,
        )

        logger.info(
            "Async компиляция feature idea завершена: title={}, specs_count={}",
            feature_idea.title,
            len(response.compiled_feature_specs),
        )
        return response.compiled_feature_specs

    async def acompile_many(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        feature_ideas: list[FeatureIdea],
    ) -> list[CompiledFeatureSpec]:
        logger.info(
            "Async компиляция нескольких идей начата: feature_ideas_count={}",
            len(feature_ideas),
        )

        tasks = [
            self.acompile_one(
                bundle=bundle,
                join_plan=join_plan,
                feature_idea=feature_idea,
            )
            for feature_idea in feature_ideas
        ]

        results = await asyncio.gather(*tasks)

        compiled_specs: list[CompiledFeatureSpec] = []
        for specs in results:
            compiled_specs.extend(specs)

        logger.info(
            "Async компиляция нескольких идей завершена: compiled_specs_count={}",
            len(compiled_specs),
        )
        return compiled_specs