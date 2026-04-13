from __future__ import annotations

import os
from pydantic import ValidationError
from collections.abc import Sequence
from typing import Any

import json
import asyncio
from typing import TypeVar

from pydantic import BaseModel


TModel = TypeVar("TModel", bound=BaseModel)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat import GigaChat
from loguru import logger

from src.common.exceptions import (
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMInvocationError,
    LLMJsonExtractionError,
    LLMJsonValidationError,
)

class GigaChatClient:
    def __init__(
        self,
        *,
        model: str = "GigaChat-2-Max",
        temperature: float = 0.0,
        timeout: int = 60,
        verify_ssl_certs: bool = False,
    ) -> None:
        credentials = os.getenv("GIGACHAT_CREDENTIALS")
        scope = os.getenv("GIGACHAT_SCOPE")

        if not credentials:
            raise LLMConfigurationError("GIGACHAT_CREDENTIALS is not set")
        if not scope:
            raise LLMConfigurationError("GIGACHAT_SCOPE is not set")

        logger.info(
            "Инициализация GigaChatClient: model={}, temperature={}, timeout={}",
            model,
            temperature,
            timeout,
        )

        self._client = GigaChat(
            credentials=credentials,
            scope=scope,
            model=model,
            temperature=temperature,
            timeout=timeout,
            verify_ssl_certs=verify_ssl_certs,
        )

    def invoke(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        messages: list[SystemMessage | HumanMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=user_prompt))

        logger.debug(
            "Вызов GigaChat: has_system_prompt={}, user_prompt_length={}",
            bool(system_prompt),
            len(user_prompt),
        )

        try:
            response = self._client.invoke(messages)
        except Exception as exc:
            logger.exception("Ошибка вызова GigaChat")
            raise LLMInvocationError("GigaChat invocation failed") from exc

        return self._extract_text_response(response)

    def _extract_text_response(self, response: Any) -> str:
        content = getattr(response, "content", None)
        if content is None:
            logger.error(
                "GigaChat вернул ответ без content, response_type={}",
                type(response).__name__,
            )
            raise LLMEmptyResponseError("GigaChat returned response without content")

        if isinstance(content, str):
            normalized = content.strip()
            if not normalized: 
                logger.error("GigaChat вернул пустую строку в content")
                raise LLMEmptyResponseError("GigaChat returned empty response")
            return normalized

        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            normalized = self._extract_text_from_sequence(content)
            if not normalized:
                logger.error(
                    "GigaChat вернул sequence content, но текст извлечь не удалось: {}",
                    content,
                )
                raise LLMEmptyResponseError("GigaChat returned unsupported sequence content")
            return normalized

        logger.error( 
            "Ожидалась строка или список текстовых блоков от GigaChat, получен тип={}",
            type(content).__name__,
        )
        raise LLMInvocationError(
            f"Expected text response from GigaChat, got {type(content).__name__}"
        )

    def _extract_text_from_sequence(self, content: Sequence[Any]) -> str:
        text_parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                normalized = item.strip()
                if normalized:
                    text_parts.append(normalized)
                continue

            if isinstance(item, dict):
                candidate = item.get("text")
                if isinstance(candidate, str):
                    normalized = candidate.strip()
                    if normalized:
                        text_parts.append(normalized)
                continue

            candidate = getattr(item, "text", None)
            if isinstance(candidate, str):
                normalized = candidate.strip()
                if normalized:
                    text_parts.append(normalized)

        return "\n".join(text_parts).strip()


class LLMClient:
    """
    Тонкая обертка для будущего переключения между провайдерами.
    Наружу всегда возвращает строго str.
    """

    def __init__(self, provider: str = "gigachat", **kwargs: Any) -> None:
        normalized = provider.lower().strip()

        if normalized != "gigachat":
            raise LLMConfigurationError(
                f"Unsupported provider: {provider}. Only 'gigachat' is supported now."
            )

        self._provider = GigaChatClient(**kwargs)

    def invoke(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        response_text = self._provider.invoke(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        if not isinstance(response_text, str):
            logger.error(
                "Провайдер вернул не строку, provider_type={}, response_type={}",
                type(self._provider).__name__,
                type(response_text).__name__,
            )
            raise LLMInvocationError(
                f"Expected string response from provider, got {type(response_text).__name__}"
            )

        normalized = response_text.strip()
        if not normalized:
            logger.error("Провайдер вернул пустую строку после нормализации")
            raise LLMEmptyResponseError("LLM provider returned empty response")

        return normalized
    
    async def ainvoke(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        return await asyncio.to_thread(self.invoke, user_prompt=user_prompt, system_prompt=system_prompt)

    def invoke_json(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        response_model: type[TModel],
    ) -> TModel:
        logger.info(
            "LLMClient.invoke_json start: provider={}, response_model={}",
            self._provider,
            response_model.__name__,
        )

        response_text = self.invoke(user_prompt=user_prompt, system_prompt=system_prompt)
        json_payload = self.extract_json_payload(response_text=response_text)

        try:
            parsed = response_model.model_validate_json(json_payload)
        except ValidationError as exc:
            logger.exception(
                "Ошибка валидации JSON-ответа LLM: response_model={}",
                response_model.__name__,
            )
            raise LLMJsonValidationError(
                f"Не удалось провалидировать JSON в {response_model.__name__}: {exc}"
            ) from exc

        logger.info(
            "LLMClient.invoke_json success: provider={}, response_model={}",
            self._provider,
            response_model.__name__,
        )
        return parsed

    async def ainvoke_json(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        response_model: type[TModel],
    ) -> TModel:
        logger.info(
            "LLMClient.ainvoke_json start: provider={}, response_model={}",
            self._provider,
            response_model.__name__,
        )

        response_text = await self.ainvoke(user_prompt=user_prompt, system_prompt=system_prompt)
        json_payload = self.extract_json_payload(response_text=response_text)

        try:
            parsed = response_model.model_validate_json(json_payload)
        except ValidationError as exc:
            logger.exception(
                "Ошибка async-валидации JSON-ответа LLM: response_model={}",
                response_model.__name__,
            )
            raise LLMJsonValidationError(
                f"Не удалось провалидировать JSON в {response_model.__name__}: {exc}"
            ) from exc

        logger.info(
            "LLMClient.ainvoke_json success: provider={}, response_model={}",
            self._provider,
            response_model.__name__,
        )
        return parsed

    def extract_json_payload(self, *, response_text: str) -> str:
        text = response_text.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").strip()

        if text.endswith("```"):
            text = text[:-3].strip()

        if self._is_valid_json(text): 
            return text

        first_brace = text.find("{")
        last_brace = text.rfind("}")

        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            raise LLMJsonExtractionError("Не удалось извлечь JSON-объект из ответа LLM")

        candidate = text[first_brace : last_brace + 1]

        if not self._is_valid_json(candidate):
            raise LLMJsonExtractionError(
                "Из ответа LLM был извлечён фрагмент, но он не является валидным JSON"
            )

        return candidate

    def _is_valid_json(self, payload: str) -> bool:
        try:
            json.loads(payload)
        except json.JSONDecodeError:
            return False
        return True