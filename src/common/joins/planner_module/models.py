from __future__ import annotations

from pydantic import BaseModel, Field


class JoinCandidate(BaseModel):
    left_table: str = Field(description="Имя левой таблицы")
    left_key: str = Field(description="Ключ в левой таблице")
    right_table: str = Field(description="Имя правой таблицы")
    right_key: str = Field(description="Ключ в правой таблице")
    is_train_related: bool = Field(
        description="Признак того, что связь напрямую затрагивает train",
    )

    def __str__(self) -> str:
        return (
            "JoinCandidate("
            f"left_table={self.left_table!r}, "
            f"left_key={self.left_key!r}, "
            f"right_table={self.right_table!r}, "
            f"right_key={self.right_key!r}, "
            f"is_train_related={self.is_train_related!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class JoinMultiplicity(BaseModel):
    left_is_unique: bool = Field(description="Ключ слева близок к уникальному")
    right_is_unique: bool = Field(description="Ключ справа близок к уникальному")
    relation_type: str = Field(
        description="Тип связи: one_to_one, one_to_many, many_to_one, many_to_many",
    )
    left_unique_ratio: float = Field(description="Доля уникальных значений ключа слева")
    right_unique_ratio: float = Field(description="Доля уникальных значений ключа справа")

    def __str__(self) -> str:
        return (
            "JoinMultiplicity("
            f"relation_type={self.relation_type!r}, "
            f"left_is_unique={self.left_is_unique!r}, "
            f"right_is_unique={self.right_is_unique!r}, "
            f"left_unique_ratio={self.left_unique_ratio:.3f}, "
            f"right_unique_ratio={self.right_unique_ratio:.3f}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class JoinEdge(BaseModel):
    parent_table: str = Field(description="Родительская таблица в join tree")
    child_table: str = Field(description="Дочерняя таблица в join tree")
    parent_key: str = Field(description="Ключ в родительской таблице")
    child_key: str = Field(description="Ключ в дочерней таблице")
    relation_type: str = Field(
        description="Тип связи между таблицами",
    )
    requires_aggregation: bool = Field(
        description="Нужно ли агрегировать дочернюю таблицу до merge",
    )
    distance_from_train: int = Field(
        description="Расстояние от train до child_table в числе ребер",
    )
    path_from_train: list[str] = Field(
        default_factory=list,
        description="Путь от train до текущей дочерней таблицы",
    )
    reason: str = Field(
        description="Краткое объяснение, почему выбрано именно это ребро",
    )

    def __str__(self) -> str:
        return (
            "JoinEdge("
            f"parent_table={self.parent_table!r}, "
            f"child_table={self.child_table!r}, "
            f"parent_key={self.parent_key!r}, "
            f"child_key={self.child_key!r}, "
            f"relation_type={self.relation_type!r}, "
            f"requires_aggregation={self.requires_aggregation!r}, "
            f"distance_from_train={self.distance_from_train}, "
            f"path_from_train={self.path_from_train!r}, "
            f"reason={self.reason!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def to_prompt(self) -> str:
        return (f"- {self.parent_table}.{self.parent_key} -> "
                    f"{self.child_table}.{self.child_key}; "
                    f"relation_type={self.relation_type}; "
                    f"requires_aggregation={self.requires_aggregation}; "
                    f"path={self.path_from_train}")


class JoinPlan(BaseModel):
    root_table: str = Field(description="Корневая таблица плана джойнов")
    edges: list[JoinEdge] = Field(
        default_factory=list,
        description="Ребра дерева джойнов",
    )
    skipped_tables: list[str] = Field(
        default_factory=list,
        description="Таблицы, которые пока не удалось безопасно встроить в дерево",
    )

    def __str__(self) -> str:
        return (
            "JoinPlan("
            f"root_table={self.root_table!r}, "
            f"edges_count={len(self.edges)}, "
            f"skipped_tables={self.skipped_tables!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class JoinValidationResponse(BaseModel):
    is_valid: bool = Field(description="План выглядит валидным с точки зрения LLM")
    comment: str | None = Field(
        default=None,
        description="Краткий комментарий по плану",
    )

    def __str__(self) -> str:
        return (
            "JoinValidationResponse("
            f"is_valid={self.is_valid!r}, "
            f"comment={self.comment!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()