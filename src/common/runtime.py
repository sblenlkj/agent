from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from src.common.constraints import MAX_RUNTIME_SEC

class RuntimeBudgetExceededError(Exception):
    pass


@dataclass(frozen=True)
class RuntimeSnapshot:
    started_at_monotonic: float
    elapsed_seconds: float
    remaining_seconds: float
    limit_seconds: float

    def __str__(self) -> str:
        return (
            "RuntimeSnapshot("
            f"elapsed_seconds={self.elapsed_seconds:.3f}, "
            f"remaining_seconds={self.remaining_seconds:.3f}, "
            f"limit_seconds={self.limit_seconds:.3f}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class RuntimeBudget:
    def __init__(self, *, limit_seconds: float = 600.0) -> None:
        self._limit_seconds = float(limit_seconds)
        self._started_at_monotonic = time.monotonic()

        logger.info(
            "Инициализирован RuntimeBudget: limit_seconds={}",
            self._limit_seconds,
        )

    @property
    def limit_seconds(self) -> float:
        return self._limit_seconds

    @property
    def started_at_monotonic(self) -> float:
        return self._started_at_monotonic

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at_monotonic

    def remaining_seconds(self) -> float:
        remaining = self._limit_seconds - self.elapsed_seconds()
        return max(0.0, remaining)

    def snapshot(self) -> RuntimeSnapshot:
        elapsed = self.elapsed_seconds()
        remaining = max(0.0, self._limit_seconds - elapsed)
        return RuntimeSnapshot(
            started_at_monotonic=self._started_at_monotonic,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            limit_seconds=self._limit_seconds,
        )

    def check(self, *, stage_name: str) -> None:
        elapsed = self.elapsed_seconds()
        remaining = self._limit_seconds - elapsed

        logger.info(
            "Проверка RuntimeBudget: stage_name='{}', elapsed_seconds={:.3f}, remaining_seconds={:.3f}",
            stage_name,
            elapsed,
            remaining,
        )

        if remaining <= 0:
            logger.error(
                "Превышен runtime budget: stage_name='{}', elapsed_seconds={:.3f}, limit_seconds={:.3f}",
                stage_name,
                elapsed,
                self._limit_seconds,
            )
            raise RuntimeBudgetExceededError(
                f"Превышен лимит выполнения. stage_name={stage_name!r}, "
                f"elapsed_seconds={elapsed:.3f}, limit_seconds={self._limit_seconds:.3f}"
            )

    def __str__(self) -> str:
        snapshot = self.snapshot()
        return (
            "RuntimeBudget("
            f"elapsed_seconds={snapshot.elapsed_seconds:.3f}, "
            f"remaining_seconds={snapshot.remaining_seconds:.3f}, "
            f"limit_seconds={snapshot.limit_seconds:.3f}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


GLOBAL_RUNTIME_BUDGET = RuntimeBudget(limit_seconds=MAX_RUNTIME_SEC-50)