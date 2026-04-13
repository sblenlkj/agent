from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.agent.runtime import get_agent_globals
from src.agent.state import AgentRunState
from src.common.feature_codegen.models import GeneratedFeatureCode
from src.common.runtime import GLOBAL_RUNTIME_BUDGET


def generate_and_apply_feature_code_node(
    state: AgentRunState,
) -> dict[str, object]:
    logger.info("Старт ноды generate_and_apply_feature_code_node")
    GLOBAL_RUNTIME_BUDGET.check(
        stage_name="generate_and_apply_feature_code_node.start"
    )

    bundle = state.get("input_bundle")
    if bundle is None:
        raise ValueError("В state отсутствует input_bundle")

    join_plan = state.get("join_plan")
    if join_plan is None:
        raise ValueError("В state отсутствует join_plan")

    feature_ideas = state.get("feature_ideas")
    if not feature_ideas:
        raise ValueError("В state отсутствуют feature_ideas")

    globals_ = get_agent_globals()

    generated_feature_codes: list[GeneratedFeatureCode] = []
    applied_feature_titles: list[str] = []

    debug_dir = Path("output") / "debug" / "agent_codegen"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for index, feature_idea in enumerate(feature_ideas, start=1):
        GLOBAL_RUNTIME_BUDGET.check(
            stage_name=f"generate_and_apply_feature_code_node.feature_idea_{index}/{len(feature_ideas)}"
        )
        if GLOBAL_RUNTIME_BUDGET.remaining_seconds() < 250:
            logger.warning(
                "BREAK INSIDE feature_idea loop. index={}, feature_ideas_count={}, feature_title={}",
                index,
                len(feature_ideas),
                feature_idea.title,
            )
            break

        logger.info(
            "Обрабатываю feature idea {} из {}: {}",
            index,
            len(feature_ideas),
            feature_idea.title,
        )
        


        try:
            generated_feature_code = globals_.feature_code_generation_service.generate_one(
                bundle=bundle,
                join_plan=join_plan,
                feature_idea=feature_idea,
            )
        except Exception:
            logger.exception(
                "Не удалось сгенерировать код для идеи: {}",
                feature_idea.title,
            )
            continue

        generated_feature_codes.append(generated_feature_code)

        code_path = debug_dir / f"generated_feature_code_{index}.json"
        try:
            code_path.write_text(
                generated_feature_code.model_dump_json(
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("Generated code сохранен: {}", code_path)
            logger.info("Сгенерированный код:\n{}", generated_feature_code.code)
        except Exception:
            logger.exception(
                "Не удалось сохранить generated code для идеи: {}",
                feature_idea.title,
            )

        try:
            globals_.generated_features_dataset_builder.apply_generated_code(
                generated_feature_code=generated_feature_code,
            )
        except Exception:
            failed_code_path = debug_dir / f"failed_generated_code_{index}.py"
            try:
                failed_code_path.write_text(
                    generated_feature_code.code,
                    encoding="utf-8",
                )
                logger.info("Упавший код сохранен: {}", failed_code_path)
            except Exception:
                logger.exception(
                    "Не удалось сохранить упавший код для идеи: {}",
                    feature_idea.title,
                )

            logger.exception(
                "Не удалось применить generated code для идеи: {}",
                feature_idea.title,
            )
            continue

        applied_feature_titles.append(generated_feature_code.title)
        logger.info(
            "Generated code успешно применен: {}",
            generated_feature_code.title,
        )

        GLOBAL_RUNTIME_BUDGET.check(
            stage_name=f"generate_and_apply_feature_code_node.after_feature_{index}"
        )

    train_df = globals_.generated_features_dataset_builder.train_df
    test_df = globals_.generated_features_dataset_builder.test_df

    logger.info(
        "Нода generate_and_apply_feature_code_node завершена: "
        "generated_codes_count={}, applied_count={}, train_shape={}x{}, test_shape={}x{}",
        len(generated_feature_codes),
        len(applied_feature_titles),
        len(train_df),
        len(train_df.columns),
        len(test_df),
        len(test_df.columns),
    )

    GLOBAL_RUNTIME_BUDGET.check(
        stage_name="generate_and_apply_feature_code_node.end"
    )

    return {
        "generated_feature_codes": generated_feature_codes,
        "applied_feature_titles": applied_feature_titles,
    }