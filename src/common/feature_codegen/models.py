from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedFeatureCode(BaseModel):
    title: str = Field(description="Название генерируемого признакового блока")
    merge_back_keys: list[str] = Field(
        ...,
        description="Ключи, по которым итоговый feature dataframe нужно мержить в train и test",
    )
    code: str = Field(
        description="Python-код с функцией build_feature(train_df, tables) -> pd.DataFrame"
    )
    notes: str | None = Field(
        default=None,
        description="Дополнительные замечания по сгенерированному коду",
    )

    def __str__(self) -> str:
        return (
            "GeneratedFeatureCode("
            f"title={self.title!r}, "
            f"merge_back_keys={self.merge_back_keys!r}, "
            f"code_len={len(self.code)!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class GeneratedFeatureCodeResponse(BaseModel):
    generated_feature_code: GeneratedFeatureCode = Field(
        description="Сгенерированный код для построения одного feature dataframe",
    )

    def __str__(self) -> str:
        return (
            "GeneratedFeatureCodeResponse("
            f"title={self.generated_feature_code.title!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()