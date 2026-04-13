from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from src.common.paths import PATHS

from src.common.feature_codegen.prepared_dataset_builder import (
    GeneratedFeaturesDatasetBuilder,
)
from src.common.feature_codegen.service import FeatureCodeGenerationService
from src.common.feature_ideas_generation.service import FeatureIdeasGenerationService
from src.common.io.repository import InputRepository
from src.common.llm_client import LLMClient
from src.common.readme.service import ReadmeService
from src.common.joins.feature_planning_v1.service import TableFeaturePlanningService
from src.common.joins.feature_planning_v1.executor import FeatureExecutor
from src.stats.scoring_v2 import CatBoostFeatureSelector


@dataclass(frozen=True)
class AgentGlobals:
    llm_client: LLMClient
    input_repository: InputRepository
    readme_service: ReadmeService
    feature_ideas_generation_service: FeatureIdeasGenerationService
    feature_code_generation_service: FeatureCodeGenerationService
    generated_features_dataset_builder: GeneratedFeaturesDatasetBuilder
    
    legacy_feature_planning_servicer: TableFeaturePlanningService
    legacy_executor: FeatureExecutor

    catboost_feature_selector: CatBoostFeatureSelector


AGENT_GLOBALS: AgentGlobals | None = None


def init_agent_globals(
    *,
    data_dir: Path,
    llm_client: LLMClient,
) -> AgentGlobals:
    global AGENT_GLOBALS

    logger.info("Инициализирую AgentGlobals: data_dir={}", data_dir)

    AGENT_GLOBALS = AgentGlobals(
        llm_client=llm_client,
        input_repository=InputRepository(data_path=data_dir),
        readme_service=ReadmeService(llm_client=llm_client),
        feature_ideas_generation_service=FeatureIdeasGenerationService(
            llm_client=llm_client,
        ),
        feature_code_generation_service=FeatureCodeGenerationService(
            llm_client=llm_client,
        ),
        generated_features_dataset_builder=GeneratedFeaturesDatasetBuilder(
            data_dir=data_dir,
        ),
        legacy_feature_planning_servicer=TableFeaturePlanningService(
            llm_client=llm_client,
        ),
        legacy_executor=FeatureExecutor(
            data_dir=PATHS.data_dir
        ),
        catboost_feature_selector=CatBoostFeatureSelector()
    )
    return AGENT_GLOBALS


def get_agent_globals() -> AgentGlobals:
    if AGENT_GLOBALS is None:
        raise RuntimeError(
            "AgentGlobals не инициализирован. "
            "Сначала вызови init_agent_globals(...)."
        )
    return AGENT_GLOBALS