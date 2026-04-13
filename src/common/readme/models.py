from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ReadmeColumnDescription(BaseModel):
    table_name: str = Field(description="Имя существующей таблицы")
    column_name: str = Field(description="Имя существующей колонки")
    description: str | None = Field(
        default=None,
        description="Краткое семантическое описание колонки",
    )
    allowed_values: str | None = Field(
        default=None,
        description="Допустимые значения колонки, если они явно указаны в README",
    )

    @field_validator("table_name", "column_name")
    @classmethod
    def validate_required_names(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Обязательное строковое поле не должно быть пустым")
        return normalized

    @field_validator("description", "allowed_values")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def __str__(self) -> str:
        return (
            "ReadmeColumnDescription("
            f"table_name={self.table_name!r}, "
            f"column_name={self.column_name!r}, "
            f"description={self.description!r}, "
            f"allowed_values={self.allowed_values!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class ReadmeTableDescription(BaseModel):
    table_name: str = Field(description="Имя существующей таблицы")
    description: str | None = Field(
        default=None,
        description="Краткое семантическое описание таблицы",
    )
    columns: list[ReadmeColumnDescription] = Field(
        default_factory=list,
        description="Список описаний колонок таблицы",
    )

    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Имя таблицы не должно быть пустым")
        return normalized

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def __str__(self) -> str:
        return (
            "ReadmeTableDescription("
            f"table_name={self.table_name!r}, "
            f"description={self.description!r}, "
            f"columns_count={len(self.columns)}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class ReadmeParseResponse(BaseModel):
    task_description: str | None = Field(
        default=None,
        description="Краткое описание ML-задачи из README",
    )
    target_column_name: str | None = Field(
        default=None,
        description="Имя целевой колонки, если оно явно следует из README",
    )
    tables: list[ReadmeTableDescription] = Field(
        default_factory=list,
        description="Список описаний известных таблиц и их колонок",
    )

    @field_validator("task_description", "target_column_name")
    @classmethod
    def normalize_optional_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def __str__(self) -> str:
        return (
            "ReadmeParseResponse("
            f"task_description={self.task_description!r}, "
            f"target_column_name={self.target_column_name!r}, "
            f"tables_count={len(self.tables)}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class ReadmeParseArtifacts(BaseModel):
    raw_response_text: str = Field(
        description="Сырой текстовый ответ LLM до извлечения JSON",
    )
    parsed_response: ReadmeParseResponse = Field(
        description="Распарсенный и провалидированный ответ README parser",
    )

    def __str__(self) -> str:
        return (
            "ReadmeParseArtifacts("
            f"raw_response_length={len(self.raw_response_text)}, "
            f"tables_count={len(self.parsed_response.tables)}, "
            f"target_column_name={self.parsed_response.target_column_name!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()