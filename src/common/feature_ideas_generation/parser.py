from __future__ import annotations

from pydantic import BaseModel, Field

from src.common.feature_ideas_generation.models import FeatureIdea


class FeatureIdeasResponse(BaseModel):
    feature_ideas: list[FeatureIdea] = Field(
        default_factory=list,
        description="Список идей признаков",
    )

    def __str__(self) -> str:
        return f"FeatureIdeasResponse(feature_ideas_count={len(self.feature_ideas)})"

    def __repr__(self) -> str:
        return self.__str__()