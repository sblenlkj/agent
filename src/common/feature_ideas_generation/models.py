from __future__ import annotations

from pydantic import BaseModel, Field

class FeatureIdea(BaseModel):
    title: str = Field(description="Краткое название идеи признаков")
    hypothesis: str = Field(
        description="Почему эта идея может быть полезна для предсказания target"
    )
    priority: int = Field(
        description="Приоритет идеи, где 1 означает самый высокий приоритет"
    )
    required_tables: list[str] = Field(
        default_factory=list,
        description="Какие таблицы нужны для реализации идеи",
    )
    required_join_paths: list[list[str]] = Field(
        default_factory=list,
        description="Какие пути джойнов нужны для реализации идеи",
    )
    candidate_feature_families: list[str] = Field(
        default_factory=list,
        description="Какие типы признаков можно построить внутри идеи",
    )
    notes: str | None = Field(
        default=None,
        description="Дополнительные замечания по идее",
    )

    def __str__(self) -> str:
        return (
            "FeatureIdea("
            f"title={self.title!r}, "
            f"priority={self.priority!r}, "
            f"required_tables={self.required_tables!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class FeatureIdeasResponse(BaseModel):
    feature_ideas: list[FeatureIdea] = Field(
        default_factory=list,
        description="Список идей признаков",
    )

    def __str__(self) -> str:
        return f"FeatureIdeasResponse(feature_ideas_count={len(self.feature_ideas)})"

    def __repr__(self) -> str:
        return self.__str__()