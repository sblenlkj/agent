from __future__ import annotations

from loguru import logger

from src.agent.state import AgentRunState
from src.agent.runtime import get_agent_globals, AgentGlobals

from src.common.runtime import GLOBAL_RUNTIME_BUDGET
from src.common.readme.input_bundle_cache import InputBundleCache

from src.common.paths import PATHS

def prepare_input_bundle_node(
    state: AgentRunState,
) -> dict[str, object]:
    logger.info("Старт ноды prepare_input_bundle")

    GLOBAL_RUNTIME_BUDGET.check(stage_name="prepare_input_bundle_node.start")

    globals_: AgentGlobals = get_agent_globals()

    logger.info("Загружаю raw InputBundle из InputRepository")
    raw_bundle = globals_.input_repository.load()

    logger.info("Запускаю README enrichment через ReadmeService")
    GLOBAL_RUNTIME_BUDGET.check(
        stage_name="prepare_input_bundle_node.readme_llm_call_start"
    )
    _, enriched_bundle = globals_.readme_service.parse_and_enrich(raw_bundle)
    GLOBAL_RUNTIME_BUDGET.check(
        stage_name="prepare_input_bundle_node.readme_llm_call_end"
    )

    logger.info("Нода prepare_input_bundle завершена успешно")

    GLOBAL_RUNTIME_BUDGET.check(stage_name="prepare_input_bundle_node.end")

    return {
        "input_bundle": enriched_bundle,
    }






def prepare_input_bundle_from_cache_node(
    state: AgentRunState,
) -> dict[str, object]:
    logger.info("Старт mock-ноды prepare_input_bundle_from_cache")

    GLOBAL_RUNTIME_BUDGET.check(
        stage_name="prepare_input_bundle_from_cache_node.start"
    )

    cache = InputBundleCache()
    cache_path = PATHS.output_dir / "cache" / "enriched_input_bundle.json"

    logger.info("Пытаюсь загрузить InputBundle из кэша: {}", cache_path)

    if not cache.exists(path=cache_path):
        raise FileNotFoundError(
            f"Кэш enriched InputBundle не найден: {cache_path}. "
            "Сначала запусти smoke_test_readme_service.py"
        )

    bundle = cache.load(path=cache_path)

    logger.info(
        "InputBundle успешно загружен из кэша: train={}, test={}, additional_tables={}",
        bundle.train.name,
        bundle.test.name,
        list(bundle.additional_tables.keys()),
    )

    GLOBAL_RUNTIME_BUDGET.check(
        stage_name="prepare_input_bundle_from_cache_node.end"
    )

    return {
        "input_bundle": bundle,
    }