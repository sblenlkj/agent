from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.common.io.exceptions import SeparatorDetectionError
from src.common.llm_client import LLMClient

COMMON_SEPARATORS = [",", ";", "\t", "|"]
CLIENT = None # LLMClient()


class FileManager:
    @staticmethod
    def read_text(path: Path) -> str:
        logger.debug("Reading text file: {}", path)
        return path.read_text(encoding="utf-8")
    

    @staticmethod
    def detect_separator_by_llm(path: Path) -> str:
        sample = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
        preview_text = "\n".join(sample)

        response = CLIENT.invoke(user_prompt=("Вот кусочек CSV-файла. Найди разделитель колонок. "
            "Верни только сам разделитель одним символом.\n\n"
            f"{preview_text}")
        )

        return str(getattr(response, "content", response)).strip()

    @staticmethod
    def detect_separator(path: Path) -> str:
        logger.debug("Detecting separator for file: {}", path)
        sample = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]

        best_sep: str | None = None
        best_score = 0

        logs = []

        for sep in COMMON_SEPARATORS:
            counts = [line.count(sep) for line in sample if line]
            if not counts:
                continue

            score = min(counts)
            if score > best_score:
                best_score = score
                best_sep = sep

        if best_sep is None:
            logger.error("Could not detect separator for file: {}", path)

            best_sep = FileManager.detect_separator_by_llm(path)

            if best_sep in COMMON_SEPARATORS:

                raise SeparatorDetectionError(f"Could not detect separator for file {path}, {sample[10:20:-1]}")

        logger.debug("Detected separator '{}' for {}", best_sep, path)
        return best_sep

    @staticmethod
    def read_csv(path: Path) -> tuple[pd.DataFrame, str]:
        sep = FileManager.detect_separator(path)
        logger.debug("Reading CSV file: {} with separator '{}'", path, sep)
        df = pd.read_csv(path, sep=sep)
        logger.debug("Loaded dataframe from {} with shape {}", path.name, df.shape)
        return df, sep

    @staticmethod
    def list_csv_files(directory: Path) -> list[Path]:
        files = sorted(directory.glob("*.csv"))
        logger.debug("Found CSV files in {}: {}", directory, [f.name for f in files])
        return files

    @staticmethod
    def write_csv(df: pd.DataFrame, path: Path) -> None:
        logger.debug("Writing dataframe to {} with shape {}", path, df.shape)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
