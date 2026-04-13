from __future__ import annotations

from pydantic import BaseModel, Field

from src.common.schema.models import ColumnState


class SourceTable(BaseModel):
    name: str = Field(description="Логическое имя таблицы")
    file_name: str = Field(description="Исходное имя файла")
    separator: str = Field(description="Определённый разделитель CSV")
    row_count: int = Field(description="Количество строк")
    column_count: int = Field(description="Количество столбцов")
    columns: list[ColumnState] = Field(
        default_factory=list,
        description="Структурированное состояние столбцов",
    )
    candidate_keys: list[str] = Field(
        default_factory=list,
        description="Потенциальные ключи для join, найденные в сырых данных",
    )
    description: str | None = Field(
        default=None,
        description="Семантическое описание таблицы",
    )

    def get_column(self, column_name: str) -> ColumnState | None:
        for column in self.columns:
            if column.name == column_name:
                return column
        return None

    def get_column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def __str__(self) -> str:
        return (
            "SourceTable("
            f"name={self.name!r}, file_name={self.file_name!r}, "
            f"rows={self.row_count}, cols={self.column_count}, "
            f"candidate_keys={self.candidate_keys}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()

class CandidateKeyMatch(BaseModel):
    """DEPRECATED: logic has been moved to join submodule!!!"""

    left_table: str = Field(description="Имя левой таблицы")
    left_key: str = Field(description="Кандидатный ключ в левой таблице")
    right_table: str = Field(description="Имя правой таблицы")
    right_key: str = Field(description="Кандидатный ключ в правой таблице")
    is_train_related: bool = Field(
        default=False,
        description="Связано ли это совпадение напрямую с таблицей train",
    )

    def to_text(self) -> str:
        suffix = " [train-related]" if self.is_train_related else ""
        return (
            f"Potential key match: "
            f"{self.left_table}.{self.left_key} == {self.right_table}.{self.right_key}"
            f"{suffix}"
        )

    def __str__(self) -> str:
        return (
            "CandidateKeyMatch("
            f"{self.left_table}.{self.left_key} -> "
            f"{self.right_table}.{self.right_key}, "
            f"is_train_related={self.is_train_related}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()

class InputBundle(BaseModel):
    readme_text: str = Field(description="Полный исходный текст README")
    train: SourceTable
    test: SourceTable
    additional_tables: dict[str, SourceTable] = Field(default_factory=dict)

    def find_candidate_key_matches(self) -> list[CandidateKeyMatch]:
        """DEPRECATED: logic has been moved to join submodule!!!"""
        train_columns = set(self.train.get_column_names())
        test_columns = set(self.test.get_column_names())

        if train_columns != test_columns:
            raise ValueError("Train and test columns must match at input stage.")

        tables: list[tuple[str, SourceTable]] = [
            ("train", self.train),
            *self.additional_tables.items(),
        ]

        matches: list[CandidateKeyMatch] = []
        for i, (left_name, left_table) in enumerate(tables):
            for right_name, right_table in tables[i + 1 :]:
                common_keys = sorted(
                    set(left_table.candidate_keys) & set(right_table.candidate_keys)
                )
                for key in common_keys:
                    matches.append(
                        CandidateKeyMatch(
                            left_table=left_name,
                            left_key=key,
                            right_table=right_name,
                            right_key=key,
                            is_train_related=("train" in {left_name, right_name}),
                        )
                    )
        return matches

    def __str__(self) -> str:
        additional = list(self.additional_tables.keys())
        return (
            "InputBundle("
            f"train={self.train.name!r}, test={self.test.name!r}, "
            f"additional_tables={additional}, "
            f"readme_len={len(self.readme_text)}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


