from __future__ import annotations

from pydantic import BaseModel, Field


class PreparedDatasetMetadata(BaseModel):
    train_rows: int = Field(description="Количество строк train после подготовки")
    train_cols: int = Field(description="Количество колонок train после подготовки")
    test_rows: int = Field(description="Количество строк test после подготовки")
    test_cols: int = Field(description="Количество колонок test после подготовки")

    def __str__(self) -> str:
        return (
            "PreparedDatasetMetadata("
            f"train_rows={self.train_rows}, "
            f"train_cols={self.train_cols}, "
            f"test_rows={self.test_rows}, "
            f"test_cols={self.test_cols}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()