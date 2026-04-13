from __future__ import annotations

from pydantic import BaseModel, Field

from src.common.archive.compilation.models import CompiledFeatureSpec


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