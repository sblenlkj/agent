from __future__ import annotations

import json
import re

from loguru import logger

from src.common.io.models import InputBundle
from src.common.llm_client import LLMClient
from src.common.readme.merger import ReadmeBundleMerger
from src.common.readme.models import (
    ReadmeParseArtifacts,
    ReadmeParseResponse,
)
from src.common.readme.exceptions import ReadmeParsingError
from src.common.readme.prompt_builder import ReadmePromptBuilder
from src.common.runtime import GLOBAL_RUNTIME_BUDGET


class ReadmeParser:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        prompt_builder: ReadmePromptBuilder | None = None,
        merger: ReadmeBundleMerger | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._prompt_builder = prompt_builder or ReadmePromptBuilder()
        self._merger = merger or ReadmeBundleMerger()

    def parse(self, bundle: InputBundle) -> ReadmeParseResponse:
        artifacts = self.parse_with_artifacts(bundle)
        return artifacts.parsed_response

    def parse_with_artifacts(self, bundle: InputBundle) -> ReadmeParseArtifacts:
        logger.info("Запускаю LLM-парсинг README")

        GLOBAL_RUNTIME_BUDGET.check(stage_name="readme_parser.before_prompt_build")

        prompt = self._prompt_builder.build(bundle)

        GLOBAL_RUNTIME_BUDGET.check(stage_name="readme_parser.before_llm_call")

        raw_response = self._llm_client.invoke(
            system_prompt=(
                "Ты извлекаешь структуру табличного датасета из README. "
                "Нужно вернуть только валидный JSON строго по заданной JSON Schema. "
                "Нельзя добавлять markdown, пояснения или комментарии."
            ),
            user_prompt=prompt,
        )

        GLOBAL_RUNTIME_BUDGET.check(stage_name="readme_parser.after_llm_call")

        logger.debug("Получен raw-ответ от LLM, длина={} символов", len(raw_response))

        payload = self._extract_json_payload(raw_response)
        parsed_response = ReadmeParseResponse.model_validate_json(payload)

        logger.info(
            "README успешно распарсен: tables_count={}, target_column_name={}",
            len(parsed_response.tables),
            parsed_response.target_column_name,
        )

        return ReadmeParseArtifacts(
            raw_response_text=raw_response,
            parsed_response=parsed_response,
        )

    def parse_and_merge(self, bundle: InputBundle) -> tuple[ReadmeParseResponse, InputBundle]:
        artifacts = self.parse_with_artifacts(bundle)
        merged_bundle = self._merger.merge(
            bundle=bundle,
            response=artifacts.parsed_response,
        )
        return artifacts.parsed_response, merged_bundle

    def _extract_json_payload(self, raw_response: str) -> str:
        text = raw_response.strip()
        if not text:
            logger.error("LLM вернула пустой ответ при парсинге README")
            raise ReadmeParsingError("Пустой ответ LLM при парсинге README")

        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```",
            text,
            flags=re.DOTALL,
        )
        if fenced_match:
            logger.debug("JSON извлечен из fenced code block")
            self._validate_json_string(fenced_match.group(1))
            return fenced_match.group(1)

        object_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if object_match:
            logger.debug("JSON извлечен из raw текста ответа")
            self._validate_json_string(object_match.group(1))
            return object_match.group(1)

        logger.error("Не удалось извлечь JSON из ответа LLM: {}", raw_response)
        raise ReadmeParsingError("Не удалось извлечь JSON из ответа LLM")

    def _validate_json_string(self, payload: str) -> None:
        try:
            json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.exception("Извлеченный JSON невалиден")
            raise ReadmeParsingError("LLM вернула невалидный JSON") from exc