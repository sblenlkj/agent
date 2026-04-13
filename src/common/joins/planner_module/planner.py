from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from loguru import logger

from src.common.io.models import InputBundle, SourceTable
from src.common.joins.planner_module.exceptions import JoinPlanningError
from src.common.joins.planner_module.models import JoinCandidate, JoinEdge, JoinMultiplicity, JoinPlan


class JoinPlanner:
    def __init__(self, *, unique_threshold: float = 0.98) -> None:
        self._unique_threshold = unique_threshold

    def build_candidates(self, bundle: InputBundle) -> list[JoinCandidate]:
        logger.info("Строю список кандидатов для join planner")

        candidates = self._find_candidate_key_matches(
            train_table=bundle.train,
            additional_tables=list(bundle.additional_tables.values()),
        )

        logger.info("Построено {} кандидатов на джойн", len(candidates))
        return candidates

    def build_plan(self, bundle: InputBundle) -> JoinPlan:
        logger.info("Начинаю построение train-centric join plan")

        candidates = self.build_candidates(bundle)
        adjacency = self._build_adjacency(candidates=candidates)
        table_by_name = self._build_table_lookup(bundle)

        visited: set[str] = {"train"}
        queue: deque[tuple[str, int, list[str]]] = deque([("train", 0, ["train"])])
        edges: list[JoinEdge] = []

        while queue:
            current_table, current_distance, current_path = queue.popleft()
            logger.debug(
                "Обрабатываю таблицу '{}' в BFS: distance={}, path={}",
                current_table,
                current_distance,
                current_path,
            )

            ranked_candidates = self._rank_candidates(
                current_table=current_table,
                candidates=adjacency.get(current_table, []),
                table_by_name=table_by_name,
            )

            for candidate in ranked_candidates:
                next_table = candidate["next_table"]
                if not isinstance(next_table, str):
                    raise JoinPlanningError("next_table должен быть строкой")

                if next_table in visited:
                    continue

                current_key = candidate["current_key"]
                next_key = candidate["next_key"]
                is_train_related = candidate["is_train_related"]

                if not isinstance(current_key, str) or not isinstance(next_key, str):
                    raise JoinPlanningError("Ключи джойна должны быть строками")
                if not isinstance(is_train_related, bool):
                    raise JoinPlanningError("is_train_related должен быть bool")

                relation = self._infer_multiplicity(
                    left_table=table_by_name[current_table],
                    left_key=current_key,
                    right_table=table_by_name[next_table],
                    right_key=next_key,
                )

                requires_aggregation = relation.relation_type in {
                    "one_to_many",
                    "many_to_many",
                }

                edge = JoinEdge(
                    parent_table=current_table,
                    child_table=next_table,
                    parent_key=current_key,
                    child_key=next_key,
                    relation_type=relation.relation_type,
                    requires_aggregation=requires_aggregation,
                    distance_from_train=current_distance + 1,
                    path_from_train=[*current_path, next_table],
                    reason=self._build_reason(
                        current_table=current_table,
                        next_table=next_table,
                        relation=relation,
                        candidate_is_train_related=is_train_related,
                    ),
                )

                edges.append(edge)
                visited.add(next_table)
                queue.append((next_table, current_distance + 1, [*current_path, next_table]))

                logger.debug(
                    "Добавлено ребро join tree: {} -> {} via {} = {}, relation_type={}, requires_aggregation={}",
                    edge.parent_table,
                    edge.child_table,
                    edge.parent_key,
                    edge.child_key,
                    edge.relation_type,
                    edge.requires_aggregation,
                )

        all_tables = {"train", *bundle.additional_tables.keys()}
        skipped_tables = sorted(all_tables - visited)

        plan = JoinPlan(
            root_table="train",
            edges=edges,
            skipped_tables=skipped_tables,
        )

        logger.info(
            "Join plan построен: edges_count={}, skipped_tables={}",
            len(plan.edges),
            plan.skipped_tables,
        )
        return plan

    def _find_candidate_key_matches(
        self,
        *,
        train_table: SourceTable,
        additional_tables: list[SourceTable],
    ) -> list[JoinCandidate]:
        matches: list[JoinCandidate] = []

        additional_by_name: dict[str, SourceTable] = {
            table.name: table for table in additional_tables
        }

        # 1. train <-> additional
        for train_key in train_table.candidate_keys:
            for additional_table in additional_tables:
                for additional_key in additional_table.candidate_keys:
                    if train_key != additional_key:
                        continue

                    matches.append(
                        JoinCandidate(
                            left_table=train_table.name,
                            left_key=train_key,
                            right_table=additional_table.name,
                            right_key=additional_key,
                            is_train_related=True,
                        )
                    )

        # 2. additional <-> additional
        additional_items = list(additional_by_name.items())
        for left_index, (left_name, left_table) in enumerate(additional_items):
            for right_name, right_table in additional_items[left_index + 1 :]:
                for left_key in left_table.candidate_keys:
                    for right_key in right_table.candidate_keys:
                        if left_key != right_key:
                            continue

                        matches.append(
                            JoinCandidate(
                                left_table=left_name,
                                left_key=left_key,
                                right_table=right_name,
                                right_key=right_key,
                                is_train_related=False,
                            )
                        )

        logger.debug("Найдены candidate key matches: {}", matches)
        return matches

    def _build_table_lookup(self, bundle: InputBundle) -> dict[str, SourceTable]:
        return {
            "train": bundle.train,
            **bundle.additional_tables,
        }

    def _build_adjacency(
        self,
        *,
        candidates: Iterable[JoinCandidate],
    ) -> dict[str, list[dict[str, str | bool]]]:
        adjacency: dict[str, list[dict[str, str | bool]]] = {}

        for candidate in candidates:
            adjacency.setdefault(candidate.left_table, []).append(
                {
                    "next_table": candidate.right_table,
                    "current_key": candidate.left_key,
                    "next_key": candidate.right_key,
                    "is_train_related": candidate.is_train_related,
                }
            )
            adjacency.setdefault(candidate.right_table, []).append(
                {
                    "next_table": candidate.left_table,
                    "current_key": candidate.right_key,
                    "next_key": candidate.left_key,
                    "is_train_related": candidate.is_train_related,
                }
            )

        return adjacency

    def _rank_candidates(
        self,
        *,
        current_table: str,
        candidates: list[dict[str, str | bool]],
        table_by_name: dict[str, SourceTable],
    ) -> list[dict[str, str | bool]]:
        def score(candidate: dict[str, str | bool]) -> tuple[int, int, int]:
            next_table = candidate["next_table"]
            current_key = candidate["current_key"]
            next_key = candidate["next_key"]
            is_train_related = candidate["is_train_related"]

            if not isinstance(next_table, str):
                raise JoinPlanningError("next_table должен быть строкой")
            if not isinstance(current_key, str) or not isinstance(next_key, str):
                raise JoinPlanningError("Ключи должны быть строками")
            if not isinstance(is_train_related, bool):
                raise JoinPlanningError("is_train_related должен быть bool")

            current_unique_ratio = self._get_unique_ratio(
                table=table_by_name[current_table],
                column_name=current_key,
            )
            next_unique_ratio = self._get_unique_ratio(
                table=table_by_name[next_table],
                column_name=next_key,
            )

            aggregation_penalty = 1
            if (
                current_unique_ratio >= self._unique_threshold
                and next_unique_ratio < self._unique_threshold
            ):
                aggregation_penalty = 0
            elif (
                current_unique_ratio < self._unique_threshold
                and next_unique_ratio >= self._unique_threshold
            ):
                aggregation_penalty = 0

            train_penalty = 0 if is_train_related else 1

            dimension_bonus = 0
            if current_table == "products" and next_table in {"aisles", "departments"}:
                dimension_bonus = -1

            return (train_penalty, aggregation_penalty, dimension_bonus)

        ranked = sorted(candidates, key=score)
        logger.debug(
            "Кандидаты для '{}' после ранжирования: {}",
            current_table,
            ranked,
        )
        return ranked

    def _infer_multiplicity(
        self,
        *,
        left_table: SourceTable,
        left_key: str,
        right_table: SourceTable,
        right_key: str,
    ) -> JoinMultiplicity:
        left_unique_ratio = self._get_unique_ratio(table=left_table, column_name=left_key)
        right_unique_ratio = self._get_unique_ratio(table=right_table, column_name=right_key)

        left_is_unique = left_unique_ratio >= self._unique_threshold
        right_is_unique = right_unique_ratio >= self._unique_threshold

        if left_is_unique and right_is_unique:
            relation_type = "one_to_one"
        elif left_is_unique and not right_is_unique:
            relation_type = "one_to_many"
        elif not left_is_unique and right_is_unique:
            relation_type = "many_to_one"
        else:
            relation_type = "many_to_many"

        relation = JoinMultiplicity(
            left_is_unique=left_is_unique,
            right_is_unique=right_is_unique,
            relation_type=relation_type,
            left_unique_ratio=left_unique_ratio,
            right_unique_ratio=right_unique_ratio,
        )

        logger.debug(
            "Определена кратность связи: {}.{} <-> {}.{} => {}",
            left_table.name,
            left_key,
            right_table.name,
            right_key,
            relation,
        )
        return relation

    def _get_unique_ratio(self, *, table: SourceTable, column_name: str) -> float:
        for column in table.columns:
            if column.name == column_name:
                return float(column.unique_ratio)

        raise JoinPlanningError(
            f"Колонка {column_name!r} не найдена в таблице {table.name!r}"
        )

    def _build_reason(
        self,
        *,
        current_table: str,
        next_table: str,
        relation: JoinMultiplicity,
        candidate_is_train_related: bool,
    ) -> str:
        train_part = (
            "связь напрямую затрагивает train"
            if candidate_is_train_related
            else "связь получена через промежуточную таблицу"
        )
        relation_part = (
            f"relation_type={relation.relation_type}, "
            f"left_unique_ratio={relation.left_unique_ratio:.3f}, "
            f"right_unique_ratio={relation.right_unique_ratio:.3f}"
        )

        if current_table == "products" and next_table in {"aisles", "departments"}:
            return f"{train_part}; выбрана dimension-ветка от products; {relation_part}"

        return f"{train_part}; {relation_part}"