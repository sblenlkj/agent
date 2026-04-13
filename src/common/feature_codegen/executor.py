from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.common.feature_codegen.exceptions import GeneratedCodeExecutionError, GeneratedCodeSecurityError
from src.common.feature_codegen.models import GeneratedFeatureCode


class FeatureCodeExecutor:
    FORBIDDEN_NAMES: set[str] = {
        "eval",
        "exec",
        "open",
        "__import__",
        "globals",
        "locals",
        "compile",
        "input",
    }

    FORBIDDEN_IMPORTS: set[str] = {
        "os",
        "sys",
        "subprocess",
        "pathlib",
        "pickle",
        "shutil",
        "socket",
        "builtins",
    }

    def execute(
        self,
        *,
        generated_feature_code: GeneratedFeatureCode,
        train_df: pd.DataFrame,
        tables: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        logger.info(
            "Исполняю generated feature code: title={}",
            generated_feature_code.title,
        )

        self._validate_code_security(generated_feature_code.code)

        globals_dict: dict[str, Any] = {
            "pd": pd,
            "np": np,
        }
        locals_dict: dict[str, Any] = {}

        try:
            exec(generated_feature_code.code, globals_dict, locals_dict)
        except Exception as exc:
            logger.exception(
                "Ошибка exec при исполнении generated code: title={}",
                generated_feature_code.title,
            )
            raise GeneratedCodeExecutionError(
                f"Не удалось выполнить code для {generated_feature_code.title!r}: {exc}"
            ) from exc

        build_feature = locals_dict.get("build_feature") or globals_dict.get("build_feature")
        if build_feature is None or not callable(build_feature):
            raise GeneratedCodeExecutionError(
                f"В generated code для {generated_feature_code.title!r} "
                "не найдена функция build_feature"
            )

        safe_tables = {
            table_name: df.copy()
            for table_name, df in tables.items()
        }

        try:
            feature_df = build_feature(train_df.copy(), safe_tables)
        except Exception as exc:
            logger.exception(
                "Ошибка вызова build_feature: title={}",
                generated_feature_code.title,
            )
            raise GeneratedCodeExecutionError(
                f"Ошибка выполнения build_feature для {generated_feature_code.title!r}: {exc}"
            ) from exc

        if not isinstance(feature_df, pd.DataFrame):
            raise GeneratedCodeExecutionError(
                f"build_feature для {generated_feature_code.title!r} "
                "вернула не pandas.DataFrame"
            )

        missing_merge_keys = [
            key
            for key in generated_feature_code.merge_back_keys
            if key not in feature_df.columns
        ]
        if missing_merge_keys:
            raise GeneratedCodeExecutionError(
                f"В feature_df отсутствуют merge_back_keys {missing_merge_keys} "
                f"для {generated_feature_code.title!r}"
            )

        if feature_df.empty:
            logger.warning(
                "build_feature вернула пустой dataframe: title={}",
                generated_feature_code.title,
            )

        logger.info(
            "Generated feature code выполнен успешно: title={}, rows={}, cols={}",
            generated_feature_code.title,
            len(feature_df),
            len(feature_df.columns),
        )
        return feature_df

    def _validate_code_security(self, code: str) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise GeneratedCodeSecurityError(
                f"Синтаксическая ошибка в generated code: {exc}"
            ) from exc

        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        if "build_feature" not in function_names:
            raise GeneratedCodeSecurityError(
                "В generated code должна быть определена функция build_feature"
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".")[0]
                    if root_name in self.FORBIDDEN_IMPORTS:
                        raise GeneratedCodeSecurityError(
                            f"Запрещённый import в generated code: {root_name}"
                        )

            if isinstance(node, ast.ImportFrom):
                module_name = (node.module or "").split(".")[0]
                if module_name in self.FORBIDDEN_IMPORTS:
                    raise GeneratedCodeSecurityError(
                        f"Запрещённый import from в generated code: {module_name}"
                    )

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.FORBIDDEN_NAMES:
                        raise GeneratedCodeSecurityError(
                            f"Запрещённый вызов в generated code: {node.func.id}"
                        )

            if isinstance(node, ast.Name):
                if node.id in self.FORBIDDEN_NAMES:
                    raise GeneratedCodeSecurityError(
                        f"Запрещённое имя в generated code: {node.id}"
                    )