from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.common.io.models import InputBundle


class InputBundleCache:
    def exists(self, *, path: Path) -> bool:
        return path.exists()

    def load(self, *, path: Path) -> InputBundle:
        logger.info("Загружаю InputBundle из кэша: {}", path)
        payload = path.read_text(encoding="utf-8")
        bundle = InputBundle.model_validate_json(payload)
        logger.info("InputBundle успешно загружен из кэша")
        return bundle

    def save(self, *, bundle: InputBundle, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Сохраняю InputBundle в кэш: {}", path)
        path.write_text(
            bundle.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("InputBundle сохранен в кэш")
        return path