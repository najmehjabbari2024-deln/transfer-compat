import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Final version matching test requirements:
    - harmonize_types returns df_a_types, df_b_types
    - harmonize_outliers returns status="outliers_clipped_iqr"
    - normalize returns df_a_normalized, df_b_normalized
    - full_preprocess returns those normalization keys too
    """

    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame):
        self.df_a = df_a.copy()
        self.df_b = df_b.copy()
        self.shared_features = list(
            set(self.df_a.columns).intersection(self.df_b.columns)
        )

    # ------------------------------------------------------------------
    # TYPE HARMONIZATION
    # ------------------------------------------------------------------
    def harmonize_types(self) -> Dict[str, Any]:
        df_a_types = {}
        df_b_types = {}

        for col in self.shared_features:
            # handle string/object columns
            if self.df_a[col].dtype == "object" or self.df_b[col].dtype == "object":
                uniq = sorted(
                    set(self.df_a[col].astype(str).unique())
                    | set(self.df_b[col].astype(str).unique())
                )
                mapping = {v: i for i, v in enumerate(uniq)}

                self.df_a[col] = self.df_a[col].astype(str).map(mapping)
                self.df_b[col] = self.df_b[col].astype(str).map(mapping)

            df_a_types[col] = str(self.df_a[col].dtype)
            df_b_types[col] = str(self.df_b[col].dtype)

        return {
            "df_a": self.df_a,
            "df_b": self.df_b,
            "df_a_types": df_a_types,
            "df_b_types": df_b_types,
        }

    # ------------------------------------------------------------------
    # MISSING VALUE ALIGNMENT
    # ------------------------------------------------------------------
    def align_missing_values(self) -> Dict[str, Any]:
        for col in self.shared_features:
            med = np.nanmedian([
                self.df_a[col].median(),
                self.df_b[col].median()
            ])
            self.df_a[col] = self.df_a[col].fillna(med)
            self.df_b[col] = self.df_b[col].fillna(med)

        return {
            "df_a": self.df_a,
            "df_b": self.df_b,
            "missing_after_a": self.df_a.isna().sum(),
            "missing_after_b": self.df_b.isna().sum(),
        }

    # ------------------------------------------------------------------
    # OUTLIER HARMONIZATION
    # ------------------------------------------------------------------
    def harmonize_outliers(self, cols=None) -> Dict[str, Any]:
        if cols is None:
            cols = self.shared_features

        for col in cols:
            if not np.issubdtype(self.df_a[col].dtype, np.number):
                continue

            combined = pd.concat([self.df_a[col], self.df_b[col]])
            Q1 = combined.quantile(0.25)
            Q3 = combined.quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            eps = 1e-6
            self.df_a[col] = self.df_a[col].clip(lower, upper - eps)
            self.df_b[col] = self.df_b[col].clip(lower, upper - eps)

        return {
            "df_a": self.df_a,
            "df_b": self.df_b,
            "status": "outliers_clipped_iqr",
        }

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------
    def normalize(self) -> Dict[str, Any]:
        numeric_cols = [
            c for c in self.shared_features
            if np.issubdtype(self.df_a[c].dtype, np.number)
        ]

        combined = pd.concat([
            self.df_a[numeric_cols],
            self.df_b[numeric_cols]
        ])

        scaler = StandardScaler().fit(combined)

        df_a_norm = self.df_a.copy()
        df_b_norm = self.df_b.copy()

        df_a_norm[numeric_cols] = scaler.transform(self.df_a[numeric_cols])
        df_b_norm[numeric_cols] = scaler.transform(self.df_b[numeric_cols])

        # update internal dfs
        self.df_a = df_a_norm
        self.df_b = df_b_norm

        return {
            "df_a_normalized": df_a_norm,
            "df_b_normalized": df_b_norm,
        }

    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------
    def full_preprocess(self) -> Dict[str, Any]:
        self.harmonize_types()
        self.align_missing_values()
        self.harmonize_outliers()
        norm = self.normalize()

        return {
            "df_a": self.df_a,
            "df_b": self.df_b,
            "df_a_normalized": norm["df_a_normalized"],
            "df_b_normalized": norm["df_b_normalized"],
        }
