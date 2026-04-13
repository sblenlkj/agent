from __future__ import annotations

from loguru import logger

from src.common.io.models import InputBundle
from src.common.joins.planner_module.models import JoinPlan
from src.common.joins.feature_planning_v1.models import TableFeaturePlan
from src.common.joins.feature_planning_v1.parser import TableFeaturePlanningParser
from src.common.llm_client import LLMClient


class TableFeaturePlanningService:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        parser: TableFeaturePlanningParser | None = None,
    ) -> None:
        self._parser = parser or TableFeaturePlanningParser(
            llm_client=llm_client,
        )

    def build_table_plan(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        source_table_name: str,
    ) -> TableFeaturePlan:
        logger.info(
            "TableFeaturePlanningService.build_table_plan started: source_table_name='{}'",
            source_table_name,
        )
        plan = self._parser.parse(
            bundle=bundle,
            join_plan=join_plan,
            source_table_name=source_table_name,
        )
        logger.info(
            "TableFeaturePlanningService.build_table_plan finished: source_table_name='{}'",
            source_table_name,
        )
        return plan

    def build_multiple_table_plans(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
        source_table_names: list[str],
    ) -> list[TableFeaturePlan]:
        logger.info(
            "TableFeaturePlanningService.build_multiple_table_plans started: source_table_names={}",
            source_table_names,
        )

        plans: list[TableFeaturePlan] = []
        for source_table_name in source_table_names:
            plan = self.build_table_plan(
                bundle=bundle,
                join_plan=join_plan,
                source_table_name=source_table_name,
            )
            plans.append(plan)

        logger.info(
            "TableFeaturePlanningService.build_multiple_table_plans finished: plans_count={}",
            len(plans),
        )
        return plans

    def discover_source_table_names(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
    ) -> list[str]:
        logger.info("Автоматически определяю таблицы для feature planning")

        available_tables = set(bundle.additional_tables.keys())
        discovered: list[str] = []

        for edge in join_plan.edges:
            child_table = edge.child_table

            if child_table not in available_tables:
                continue

            if child_table == "data_dictionary":
                continue

            # dimension-таблицы второго уровня пока не спрашиваем отдельно,
            # они могут использоваться через direct_join родительской таблицы products
            if child_table in {"aisles", "departments"}:
                continue

            if child_table not in discovered:
                discovered.append(child_table)

        logger.info("Найдены source tables для feature planning: {}", discovered)
        return discovered

    def build_discovered_table_plans(
        self,
        *,
        bundle: InputBundle,
        join_plan: JoinPlan,
    ) -> list[TableFeaturePlan]:
        source_table_names = self.discover_source_table_names(
            bundle=bundle,
            join_plan=join_plan,
        )
        return self.build_multiple_table_plans(
            bundle=bundle,
            join_plan=join_plan,
            source_table_names=source_table_names,
        )