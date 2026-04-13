from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.common.io.models import InputBundle
from src.common.joins.simple_executor import JoinExecutor
from src.common.joins.planner_module.models import JoinPlan
from src.common.joins.planner_module.planner import JoinPlanner


class JoinService:
    def __init__(
        self,
        *,
        planner: JoinPlanner | None = None,
        executor: JoinExecutor | None = None,
    ) -> None:
        self._planner = planner or JoinPlanner()
        self._executor = executor

    def build_plan(self, bundle: InputBundle) -> JoinPlan:
        logger.info("JoinService.build_plan started")
        plan = self._planner.build_plan(bundle)
        logger.info("JoinService.build_plan finished")
        return plan

    def build_plan_and_execute(
        self,
        *,
        bundle: InputBundle,
        data_dir: Path,
    ) -> tuple[JoinPlan, pd.DataFrame, pd.DataFrame]:
        logger.info("JoinService.build_plan_and_execute started")

        plan = self._planner.build_plan(bundle)
        executor = self._executor or JoinExecutor(data_dir=data_dir)

        train_df, test_df = executor.execute(
            bundle=bundle,
            plan=plan,
        )

        logger.info("JoinService.build_plan_and_execute finished")
        return plan, train_df, test_df