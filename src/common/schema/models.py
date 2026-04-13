from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnState(BaseModel):
    name: str = Field(description="Имя столбца")
    dtype: str = Field(description="Тип данных Pandas")
    non_null_ratio: float = Field(description="Доля непустых значений")
    unique_ratio: float = Field(description="Доля уникальных значений")
    description: str | None = Field(default=None, description="Семантическое описание")
    importance: str | None = Field(default=None, description="Необязательная подсказка о важности")
    allowed_values: str | None = Field(
        default=None,
        description="Допустимые значения, если известны из README или словаря",
    )

    def __str__(self) -> str:
        return (
            "ColumnState("
            f"name={self.name!r}, dtype={self.dtype!r}, "
            f"non_null_ratio={self.non_null_ratio:.3f}, "
            f"unique_ratio={self.unique_ratio:.3f}, "
            f"description={self.description!r}, "
            f"importance={self.importance!r}, "
            f"allowed_values={self.allowed_values!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()
