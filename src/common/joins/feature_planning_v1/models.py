from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class FeatureAction(BaseModel):
    action_type: str = Field(
        description=(
            "Тип действия над таблицей. "
            "Допустимые значения: direct_join, aggregate_numeric, aggregate_categorical, skip"
        )
    )
    source_table: str = Field(description="Имя исходной таблицы")
    join_key: str = Field(description="Ключ таблицы, по которому признаки будут привязаны к train")
    columns: list[str] = Field(
        default_factory=list,
        description="Колонки, которые нужно использовать для данного действия",
    )
    aggregations: list[str] = Field(
        default_factory=list,
        description="Список агрегаций для данного действия",
    )
    top_k: int | None = Field(
        default=None,
        description="Размер top-k для категориальных признаков, если нужен",
    )
    reason: str = Field(
        description="Краткое объяснение, почему выбрано это действие",
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, value: str) -> str:
        normalized = value.strip()
        allowed = {
            "direct_join",
            "aggregate_numeric",
            "aggregate_categorical",
            "skip",
        }
        if normalized not in allowed:
            raise ValueError(f"Недопустимый action_type: {normalized!r}")
        return normalized

    @field_validator("source_table", "join_key", "reason")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Обязательное строковое поле не должно быть пустым")
        return normalized

    @field_validator("columns", "aggregations")
    @classmethod
    def normalize_str_list(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            normalized = item.strip()
            if normalized:
                result.append(normalized)
        return result

    def __str__(self) -> str:
        return (
            "FeatureAction("
            f"action_type={self.action_type!r}, "
            f"source_table={self.source_table!r}, "
            f"join_key={self.join_key!r}, "
            f"columns={self.columns!r}, "
            f"aggregations={self.aggregations!r}, "
            f"top_k={self.top_k!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class TableFeaturePlan(BaseModel):
    source_table: str = Field(description="Таблица, для которой строится план признаков")
    join_path: list[str] = Field(
        default_factory=list,
        description="Путь от train до таблицы",
    )
    parent_table: str | None = Field(
        default=None,
        description="Родительская таблица в дереве джойнов",
    )
    join_key: str = Field(
        description="Ключ текущей таблицы, по которому признаки будут присоединяться к train или к промежуточной ветке",
    )
    relation_type: str = Field(
        description="Тип связи между родительской таблицей и текущей таблицей",
    )
    requires_aggregation: bool = Field(
        description="Нужно ли агрегировать таблицу перед merge",
    )
    actions: list[FeatureAction] = Field(
        default_factory=list,
        description="Список действий по извлечению признаков",
    )
    comment: str | None = Field(
        default=None,
        description="Краткий комментарий по таблице",
    )

    @field_validator("source_table", "join_key", "relation_type")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Обязательное строковое поле не должно быть пустым")
        return normalized

    @field_validator("join_path")
    @classmethod
    def normalize_join_path(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            normalized = item.strip()
            if normalized:
                result.append(normalized)
        return result

    @field_validator("comment")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def __str__(self) -> str:
        return (
            "TableFeaturePlan("
            f"source_table={self.source_table!r}, "
            f"join_key={self.join_key!r}, "
            f"relation_type={self.relation_type!r}, "
            f"requires_aggregation={self.requires_aggregation!r}, "
            f"actions_count={len(self.actions)}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()