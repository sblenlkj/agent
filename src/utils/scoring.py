"""
Scoring engine: trains CatBoost on participant-generated features,
evaluates on test set using hidden labels, computes ROC-AUC and Gini.
Hidden labels (test_labels.csv) are stored separately and never
exposed to participant containers.
"""

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from app.core.logging import get_logger
from app.models.contest import Contest
from app.services.runner import DockerRunner

logger = get_logger(__name__)

CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": 1,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}


@dataclass
class ScoringResult:
    roc_auc: float = 0.0
    gini: float = 0.0
    primary_score: float = 0.0
    details: dict = field(default_factory=dict)


class ScoringEngine:
    def __init__(self, contest: Contest):
        self.contest = contest
        self.metric = contest.settings.scoring_metric if contest.settings else "roc_auc"
        self.target_column = (
            contest.settings.target_column if contest.settings else "target"
        )
        self.id_column = contest.id_column

    def score(self, output_dir: str) -> ScoringResult:
        train_df = pd.read_csv(os.path.join(output_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(output_dir, "test.csv"))

        feature_cols = [
            c for c in train_df.columns if c not in (self.id_column, self.target_column)
        ]

        X_train = train_df[feature_cols].copy()
        y_train = train_df[self.target_column].copy()
        X_test = test_df[feature_cols].copy()

        cat_features = []
        for i, col in enumerate(X_train.columns):
            if X_train[col].dtype == "object":
                X_train[col] = X_train[col].fillna("missing").astype(str)
                X_test[col] = X_test[col].fillna("missing").astype(str)
                cat_features.append(i)
            else:
                X_train[col] = X_train[col].fillna(-999)
                X_test[col] = X_test[col].fillna(-999)

        hidden_labels = self._load_hidden_labels(test_df[self.id_column])

        fit_params = {"cat_features": cat_features} if cat_features else {}

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(X_train, y_train, **fit_params)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            fold_model = CatBoostClassifier(**CATBOOST_PARAMS)
            fold_model.fit(
                X_train.iloc[train_idx], y_train.iloc[train_idx], **fit_params
            )
            probas = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
            cv_scores.append(roc_auc_score(y_train.iloc[val_idx], probas))
        cv_scores = np.array(cv_scores)

        test_probas = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(hidden_labels, test_probas)
        gini = 2 * roc_auc - 1

        feature_importance = dict(
            zip(feature_cols, model.get_feature_importance().tolist(), strict=False)
        )
        top_features = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        )

        primary_score = roc_auc if self.metric == "roc_auc" else gini

        result = ScoringResult(
            roc_auc=round(roc_auc, 6),
            gini=round(gini, 6),
            primary_score=round(primary_score, 6),
            details={
                "cv_mean_auc": round(float(cv_scores.mean()), 6),
                "cv_std_auc": round(float(cv_scores.std()), 6),
                "n_features": len(feature_cols),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "top_features": top_features,
            },
        )

        logger.info(
            "Scoring completed",
            roc_auc=result.roc_auc,
            gini=result.gini,
            cv_mean=result.details["cv_mean_auc"],
            n_features=len(feature_cols),
        )

        return result

    def ensure_target(self, output_dir: str, data_dir: str):
        """If target column missing from train.csv output, join it from source train.csv."""
        train_path = os.path.join(output_dir, "train.csv")
        train_df = pd.read_csv(train_path)
        if self.target_column in train_df.columns:
            return
        source_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
        if (
            self.target_column in source_train.columns
            and self.id_column in source_train.columns
        ):
            target_map = source_train[[self.id_column, self.target_column]]
            train_df = train_df.merge(target_map, on=self.id_column, how="left")
            train_df.to_csv(train_path, index=False)

    def _load_hidden_labels(self, test_ids: pd.Series) -> np.ndarray:
        """
        Load hidden test labels from a secure directory
        that is never mounted into participant containers.
        """
        labels_dir = DockerRunner.get_labels_dir(self.contest)
        labels_path = os.path.join(labels_dir, "test_labels.csv")

        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            merged = pd.DataFrame({self.id_column: test_ids}).merge(
                labels_df, on=self.id_column, how="left"
            )
            if self.target_column in merged.columns:
                return merged[self.target_column].values

        logger.warning(
            "Hidden labels not found, using random labels for development",
            labels_dir=labels_dir,
        )
        rng = np.random.RandomState(42)
        return rng.randint(0, 2, size=len(test_ids))
