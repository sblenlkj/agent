from __future__ import annotations

import sys

from loguru import logger

from src.common.paths import PATHS


def setup_logging(
    *,
    level: str = "INFO",
    log_to_file: bool = True,
    logs_dir_name: str = "logs",
    enqueue: bool = True,
) -> None:
    """
    Configure loguru once for the whole project.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=level.upper(),
        enqueue=enqueue,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    if log_to_file:
        logs_dir = PATHS.root / logs_dir_name
        logs_dir.mkdir(parents=True, exist_ok=True)

        logger.add(
            logs_dir / "app.log",
            level=level.upper(),
            enqueue=True,
            rotation="5 MB",
            retention=5,
            compression="zip",
            backtrace=True,
            diagnose=False,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
        )


def get_logger(name: str):
    return logger.bind(component=name)