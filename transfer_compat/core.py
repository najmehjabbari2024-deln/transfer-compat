from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from transfer_compat.metrics.shift import (
    population_stability_index,
    ks_statistic,
    jensen_shannon_divergence
)


class DatasetComparator:
    """
    Core class for analyzing dataset compatibility,
    feature alignment, and calculating transferability scores.
    """

    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame):
        self.df_a = df_a.copy()
        self.df_b = df_b.copy()

    # ----------------------------------------------------
    # 1) Basic structural comparison
    # ----------------------------------------------------
    def shape_comparison(self) -> Dict[str, Any]:
        return {
            "dataset_a_rows": self.df_a.shape[0],
            "dataset_b_rows": self.df_b.shape[0],
            "dataset_a_columns": self.df_a.shape[1],
            "dataset_b_columns": self.df_b.shape[1],
            "same_number_of_columns": self.df_a.shape[1] == self.df_b.shape[1],
            "shared_columns": list(self.shared_features())
        }

    def shared_features(self) -> List[str]:
        return list(set(self.df_a.columns).intersection(self.df_b.columns))

    # ----------------------------------------------------
    # 2) Feature Alignment
    # ----------------------------------------------------
    def feature_alignment(
        self,
        standardize: bool = True,
        drop_unshared: bool = True
    ) -> Dict[str, Any]:
        """
        Align features between source and target datasets.
        - Keeps only shared columns (optional)
        - Standardizes numeric features (optional)
        """
        shared = self.shared_features()

        df_a_aligned = self.df_a[shared].copy()
        df_b_aligned = self.df_b[shared].copy()

        if standardize:
            for col in shared:
                if pd.api.types.is_numeric_dtype(df_a_aligned[col]):
                    mean = df_a_aligned[col].mean()
                    std = df_a_aligned[col].std() or 1.0
                    df_a_aligned[col] = (df_a_aligned[col] - mean) / std
                    df_b_aligned[col] = (df_b_aligned[col] - mean) / std

        return {
            "shared_features": shared,
            "df_a_aligned": df_a_aligned,
            "df_b_aligned": df_b_aligned,
            "standardized": standardize
        }

    # ----------------------------------------------------
    # 3) Domain Adaptation Score
    # ----------------------------------------------------
    def domain_adaptation_score(
        self,
        bins: int = 10
    ) -> Dict[str, float]:
        """
        Compute domain adaptation quality using statistical drift metrics:
        - KS statistic
        - Jensen-Shannon divergence
        - Population Stability Index
        """
        shared = self.shared_features()

        ks_scores = []
        js_scores = []
        psi_scores = []

        for col in shared:
            a = self.df_a[col].dropna()
            b = self.df_b[col].dropna()

            if not pd.api.types.is_numeric_dtype(a):
                continue

            ks_scores.append(ks_statistic(a, b)["statistic"])
            js_scores.append(jensen_shannon_divergence(a, b))
            psi_scores.append(population_stability_index(a, b, bins=bins))

        # Aggregate
        return {
            "mean_ks": float(np.mean(ks_scores)) if ks_scores else 0.0,
            "mean_js": float(np.mean(js_scores)) if js_scores else 0.0,
            "mean_psi": float(np.mean(psi_scores)) if psi_scores else 0.0,
        }

    # ----------------------------------------------------
    # 4) Transferability Index (0 → 1)
    # ----------------------------------------------------
    def transferability_index(
        self,
        alpha_ks: float = 0.4,
        alpha_js: float = 0.3,
        alpha_psi: float = 0.3
    ) -> float:
        """
        Weighted score summarizing how well a model trained on dataset A
        will transfer to dataset B.
        Lower drift → higher transferability.
        """
        scores = self.domain_adaptation_score()

        # Normalize drift metrics to "compatibility"
        ks_comp = 1 - min(scores["mean_ks"], 1.0)
        js_comp = 1 - min(scores["mean_js"], 1.0)
        psi_comp = 1 - np.tanh(scores["mean_psi"])

        transfer_score = (
            alpha_ks * ks_comp +
            alpha_js * js_comp +
            alpha_psi * psi_comp
        )

        return float(max(0.0, min(1.0, transfer_score)))  # clamp 0–1
