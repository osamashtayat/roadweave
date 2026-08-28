"""Two-stage RoadWeave policy classifier with sklearn-compatible inference."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier


class HierarchicalPolicyClassifier(ClassifierMixin, BaseEstimator):
    """Separate rare lateral maneuvers from longitudinal speed decisions."""

    POLICY_CLASSES = np.asarray(
        ["KEEP", "ACCELERATE", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"],
        dtype=object,
    )
    LONGITUDINAL_CLASSES = ("KEEP", "ACCELERATE", "DECELERATE")
    NO_CHANGE = "NO_CHANGE"

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 250,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        random_state: int = 2026,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def _new_model(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            class_weight="balanced",
            early_stopping=False,
            random_state=self.random_state,
        )

    def fit(self, x_values: Any, y_values: Any, sample_weight: Optional[np.ndarray] = None):
        targets = np.asarray(y_values, dtype=object)
        lateral_targets = np.where(
            np.isin(targets, ("CHANGE_LEFT", "CHANGE_RIGHT")),
            targets,
            self.NO_CHANGE,
        )
        self.lateral_model_ = self._new_model()
        self.lateral_model_.fit(x_values, lateral_targets, sample_weight=sample_weight)

        longitudinal_mask = np.isin(targets, self.LONGITUDINAL_CLASSES)
        if not np.any(longitudinal_mask):
            raise ValueError("Policy training contains no longitudinal examples.")
        longitudinal_weights = (
            sample_weight[longitudinal_mask]
            if sample_weight is not None
            else None
        )
        self.longitudinal_model_ = self._new_model()
        self.longitudinal_model_.fit(
            x_values.iloc[longitudinal_mask]
            if hasattr(x_values, "iloc")
            else x_values[longitudinal_mask],
            targets[longitudinal_mask],
            sample_weight=longitudinal_weights,
        )
        self.classes_ = self.POLICY_CLASSES.copy()
        return self

    def predict_proba(self, x_values: Any) -> np.ndarray:
        lateral = self.lateral_model_.predict_proba(x_values)
        longitudinal = self.longitudinal_model_.predict_proba(x_values)
        lateral_map = {str(label): index for index, label in enumerate(self.lateral_model_.classes_)}
        longitudinal_map = {
            str(label): index for index, label in enumerate(self.longitudinal_model_.classes_)
        }
        output = np.zeros((len(x_values), len(self.classes_)), dtype=np.float64)
        no_change_probability = lateral[:, lateral_map[self.NO_CHANGE]]
        for output_index, label in enumerate(self.classes_):
            if label in self.LONGITUDINAL_CLASSES:
                index = longitudinal_map.get(str(label))
                if index is not None:
                    output[:, output_index] = no_change_probability * longitudinal[:, index]
            else:
                index = lateral_map.get(str(label))
                if index is not None:
                    output[:, output_index] = lateral[:, index]
        row_sums = output.sum(axis=1, keepdims=True)
        return output / np.maximum(row_sums, 1e-12)

    def predict(self, x_values: Any) -> np.ndarray:
        probabilities = self.predict_proba(x_values)
        return self.classes_[np.argmax(probabilities, axis=1)]
