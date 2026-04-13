from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.common.readme.models import ReadmeParseArtifacts


class ReadmeArtifactsWriter:
    def write(
        self,
        *,
        artifacts: ReadmeParseArtifacts,
        output_dir: Path,
        file_stem: str,
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_response_path = output_dir / f"{file_stem}_raw_response.txt"
        parsed_response_path = output_dir / f"{file_stem}_parsed_response.json"

        raw_response_path.write_text(artifacts.raw_response_text, encoding="utf-8")
        parsed_response_path.write_text(
            json.dumps(
                artifacts.parsed_response.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Артефакты README сохранены: raw='{}', parsed='{}'",
            raw_response_path,
            parsed_response_path,
        )

        return raw_response_path, parsed_response_path