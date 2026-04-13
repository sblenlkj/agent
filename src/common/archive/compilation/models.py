from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AggregationSpec(BaseModel):
    source_column: str = Field(description="Исходная колонка для агрегации")
    operation: Literal[
        "count",
        "nunique",
        "sum",
        "mean",
        "min",
        "max",
        "std",
        "ratio",
        "mode_ratio",
    ] = Field(description="Тип агрегации")
    group_by: list[str] = Field(
        default_factory=list,
        description="Список колонок группировки",
    )
    feature_name: str = Field(description="Имя итогового признака")
    filter_expression: str | None = Field(
        default=None,
        description="Опциональное текстовое условие фильтрации",
    )

    def __str__(self) -> str:
        return (
            "AggregationSpec("
            f"source_column={self.source_column!r}, "
            f"operation={self.operation!r}, "
            f"group_by={self.group_by!r}, "
            f"feature_name={self.feature_name!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class CompiledFeatureSpec(BaseModel):
    title: str = Field(description="Краткое название исполняемой спецификации")
    hypothesis: str = Field(
        description="Какой сигнал должна отражать эта спецификация"
    )
    required_tables: list[str] = Field(
        default_factory=list,
        description="Какие таблицы нужны для выполнения этой спецификации",
    )
    join_paths: list[list[str]] = Field(
        default_factory=list,
        description="Какие пути джойнов должны использоваться",
    )
    aggregations: list[AggregationSpec] = Field(
        default_factory=list,
        description="Список агрегаций, которые нужно выполнить",
    )
    notes: str | None = Field(
        default=None,
        description="Дополнительные замечания по исполнению",
    )

    def __str__(self) -> str:
        return (
            "CompiledFeatureSpec("
            f"title={self.title!r}, "
            f"required_tables={self.required_tables!r}, "
            f"aggregations={len(self.aggregations)!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class CompiledFeatureSpecsResponse(BaseModel):
    compiled_feature_specs: list[CompiledFeatureSpec] = Field(
        default_factory=list,
        description="Список исполняемых спецификаций признаков",
    )

    def __str__(self) -> str:
        return (
            "CompiledFeatureSpecsResponse("
            f"compiled_feature_specs_count={len(self.compiled_feature_specs)}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()