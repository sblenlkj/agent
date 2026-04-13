from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.common.paths import PATHS


class CatBoostSelectionResult(BaseModel):
    validation_auc_all_features: float = Field(
        description="ROC-AUC на валидации для полного числового набора признаков",
    )
    validation_auc_top5_features: float = Field(
        description="ROC-AUC на валидации для top-5 числовых признаков",
    )
    selected_feature_names: list[str] = Field(
        default_factory=list,
        description="Названия выбранных top-5 признаков",
    )
    train_csv_path: str = Field(
        description="Путь к итоговому train CSV",
    )
    test_csv_path: str = Field(
        description="Путь к итоговому test CSV",
    )

    def __str__(self) -> str:
        return (
            "CatBoostSelectionResult("
            f"validation_auc_all_features={self.validation_auc_all_features:.6f}, "
            f"validation_auc_top5_features={self.validation_auc_top5_features:.6f}, "
            f"selected_feature_names={self.selected_feature_names!r}"
            ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class CatBoostFeatureSelector:
    def __init__(
        self,
        *,
        output_dir: Path | None = None,
        raw_train_path: Path | None = None,
        raw_test_path: Path | None = None,
        random_state: int = 42,
    ) -> None:
        self._output_dir = output_dir or PATHS.output_dir
        self._raw_train_path = raw_train_path or PATHS.data_dir / "train.csv"
        self._raw_test_path = raw_test_path or PATHS.data_dir / "test.csv"
        self._random_state = random_state


    def run(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> CatBoostSelectionResult:
        logger.info(
            "Запускаю упрощённый CatBoostFeatureSelector: train_shape={}x{}, test_shape={}x{}",
            len(train_df),
            len(train_df.columns),
            len(test_df),
            len(test_df.columns),
        )

        raw_train_df = pd.read_csv(self._raw_train_path, nrows=5)
        raw_test_df = pd.read_csv(self._raw_test_path, nrows=5)

        train_base_columns = raw_train_df.columns.tolist()
        test_base_columns = raw_test_df.columns.tolist()

        logger.info("Базовые raw train колонки: {}", train_base_columns)
        logger.info("Базовые raw test колонки: {}", test_base_columns)

        self._validate_input(
            train_df=train_df,
            test_df=test_df,
            train_base_columns=train_base_columns,
            test_base_columns=test_base_columns,
        )

        numeric_feature_columns = self._select_numeric_feature_columns(
            train_df=train_df,
            train_base_columns=train_base_columns,
        )
        if not numeric_feature_columns:
            raise ValueError("Не найдено ни одной числовой feature-колонки для обучения")

        logger.info(
            "Числовые признаки для обучения: count={}, names={}",
            len(numeric_feature_columns),
            numeric_feature_columns,
        )

        if "target" not in train_df.columns:
            raise ValueError("В train_df отсутствует target, невозможно обучить CatBoost")

        X = train_df[numeric_feature_columns].copy()
        y = train_df["target"].copy()

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self._random_state,
            stratify=y,
        )

        model_all = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=self._random_state,
            verbose=False,
        )

        train_pool = Pool(X_train, y_train)
        valid_pool = Pool(X_valid, y_valid)

        logger.info(
            "Обучаю CatBoost на полном наборе числовых признаков: features_count={}",
            len(numeric_feature_columns),
        )
        model_all.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
        )

        valid_pred_all = model_all.predict_proba(valid_pool)[:, 1]
        validation_auc_all_features = roc_auc_score(y_valid, valid_pred_all)

        feature_importance_df = pd.DataFrame(
            {
                "feature_name": numeric_feature_columns,
                "importance": model_all.get_feature_importance(train_pool),
            }
        ).sort_values("importance", ascending=False, ignore_index=True)

        selected_feature_names = feature_importance_df.head(5)["feature_name"].tolist()
        if not selected_feature_names:
            raise ValueError("Не удалось выбрать top-5 признаков")

        logger.info("Top-5 числовых признаков: {}", selected_feature_names)

        model_top5 = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=self._random_state,
            verbose=False,
        )

        X_train_top5 = X_train[selected_feature_names].copy()
        X_valid_top5 = X_valid[selected_feature_names].copy()

        train_pool_top5 = Pool(X_train_top5, y_train)
        valid_pool_top5 = Pool(X_valid_top5, y_valid)

        logger.info("Обучаю CatBoost на top-5 числовых признаках")
        model_top5.fit(
            train_pool_top5,
            eval_set=valid_pool_top5,
            use_best_model=True,
        )

        valid_pred_top5 = model_top5.predict_proba(valid_pool_top5)[:, 1]
        validation_auc_top5_features = roc_auc_score(y_valid, valid_pred_top5)

        self._output_dir.mkdir(parents=True, exist_ok=True)
    
        train_csv_path = self._output_dir / "train.csv"
        test_csv_path = self._output_dir / "test.csv"

        final_train_columns = [
            *train_base_columns,
            *[col for col in selected_feature_names if col not in train_base_columns],
        ]
        final_test_columns = [
            *test_base_columns,
            *[col for col in selected_feature_names if col not in test_base_columns],
        ]

        missing_train_columns = [
            col for col in final_train_columns
            if col not in train_df.columns
        ]
        if missing_train_columns:
            raise ValueError(
                f"В train_df отсутствуют колонки для финального сохранения: {missing_train_columns}"
            )

        missing_test_columns = [
            col for col in final_test_columns
            if col not in test_df.columns
        ]
        if missing_test_columns:
            raise ValueError(
                f"В test_df отсутствуют колонки для финального сохранения: {missing_test_columns}"
            )

        final_train_df = train_df[final_train_columns].copy()
        final_test_df = test_df[final_test_columns].copy()

        final_train_df.to_csv(train_csv_path, index=False)
        final_test_df.to_csv(test_csv_path, index=False)

        logger.info(
            "Итоговые таблицы сохранены: train_path={}, test_path={}",
            train_csv_path,
            test_csv_path,
        )

        result = CatBoostSelectionResult(
            validation_auc_all_features=float(validation_auc_all_features),
            validation_auc_top5_features=float(validation_auc_top5_features),
            selected_feature_names=selected_feature_names,
            train_csv_path=str(train_csv_path),
            test_csv_path=str(test_csv_path),
        )

        logger.info("CatBoostFeatureSelector завершён: {}", result)
        return result

    def _select_numeric_feature_columns(
        self,
        *,
        train_df: pd.DataFrame,
        train_base_columns: list[str],
    ) -> list[str]:
        candidate_feature_columns = [
            column_name
            for column_name in train_df.columns
            if column_name not in train_base_columns
        ]

        numeric_feature_columns: list[str] = []
        dropped_non_numeric_columns: list[str] = []

        for column_name in candidate_feature_columns:
            if pd.api.types.is_numeric_dtype(train_df[column_name]):
                numeric_feature_columns.append(column_name)
            else:
                dropped_non_numeric_columns.append(column_name)

        if dropped_non_numeric_columns:
            logger.info(
                "Исключаю нечисловые признаки из CatBoost: {}",
                dropped_non_numeric_columns,
            )

        return numeric_feature_columns

    def _validate_input(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_base_columns: list[str],
        test_base_columns: list[str],
    ) -> None:
        missing_train_base_columns = [
            column_name
            for column_name in train_base_columns
            if column_name not in train_df.columns
        ]
        if missing_train_base_columns:
            raise ValueError(
                f"В train_df отсутствуют исходные базовые колонки: {missing_train_base_columns}"
            )

        missing_test_base_columns = [
            column_name
            for column_name in test_base_columns
            if column_name not in test_df.columns
        ]
        if missing_test_base_columns:
            raise ValueError(
                f"В test_df отсутствуют исходные базовые колонки: {missing_test_base_columns}"
            )

        if len(train_df) == 0:
            raise ValueError("train_df пуст")
        if len(test_df) == 0:
            raise ValueError("test_df пуст")