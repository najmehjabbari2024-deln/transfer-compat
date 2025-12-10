import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


class Visualizer:

    def __init__(self, df_a, df_b, shared_features=None):
        self.df_a = df_a
        self.df_b = df_b
        if shared_features is None:
            shared_features = [c for c in df_a.columns if c in df_b.columns]
        self.shared_features = shared_features

    # --------------------------------------------------------------------
    # 1) DISTRIBUTION COMPARISON
    # --------------------------------------------------------------------
    def compare_distributions(self, bins=20):
        figs = {}

        for col in self.shared_features:

            fig, ax = plt.subplots(figsize=(6, 4))

            if pd.api.types.is_numeric_dtype(self.df_a[col]):
                # Use pure matplotlib (no seaborn → avoids pandas option errors)
                a = self.df_a[col].dropna()
                b = self.df_b[col].dropna()
                ax.hist(a, bins=bins, alpha=0.5, label="A", density=True)
                ax.hist(b, bins=bins, alpha=0.5, label="B", density=True)

            else:
                a_counts = self.df_a[col].value_counts(normalize=True)
                b_counts = self.df_b[col].value_counts(normalize=True)

                idx = sorted(set(a_counts.index).union(b_counts.index))
                a_vals = [a_counts.get(i, 0) for i in idx]
                b_vals = [b_counts.get(i, 0) for i in idx]

                x = np.arange(len(idx))
                ax.bar(x - 0.2, a_vals, width=0.4, alpha=0.7, label="A")
                ax.bar(x + 0.2, b_vals, width=0.4, alpha=0.7, label="B")
                ax.set_xticks(x)
                ax.set_xticklabels(idx)

            ax.set_title(f"Distribution comparison: {col}")
            ax.legend()
            figs[col] = fig

        return {"status": "ok", "figures": figs}

    # --------------------------------------------------------------------
    # Helper: categorical drift (L1 distance)
    # --------------------------------------------------------------------
    def categorical_drift(self, col):
        a = self.df_a[col].value_counts(normalize=True)
        b = self.df_b[col].value_counts(normalize=True)
        idx = sorted(set(a.index).union(b.index))
        return sum(abs(a.get(i, 0) - b.get(i, 0)) for i in idx)

    # --------------------------------------------------------------------
    # Helper: categorical overlap
    # --------------------------------------------------------------------
    def categorical_overlap(self, col):
        a = self.df_a[col].value_counts(normalize=True)
        b = self.df_b[col].value_counts(normalize=True)
        idx = sorted(set(a.index).union(b.index))
        return sum(min(a.get(i, 0), b.get(i, 0)) for i in idx)

    # --------------------------------------------------------------------
    # 2) WASSERSTEIN TREND
    # --------------------------------------------------------------------
    def wasserstein_trend(self):
        distances = {}

        for col in self.shared_features:

            if pd.api.types.is_numeric_dtype(self.df_a[col]):
                a = self.df_a[col].dropna()
                b = self.df_b[col].dropna()
                d = wasserstein_distance(a, b)
            else:
                d = self.categorical_drift(col)

            distances[col] = float(d)

        # simple figure (required by test)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(distances.keys()), list(distances.values()))
        ax.set_title("Wasserstein / Drift Trend")

        return {"status": "ok", "distances": distances, "figure": fig}

    # --------------------------------------------------------------------
    # 3) DRIFT RADAR
    # --------------------------------------------------------------------
    def drift_radar(self):

        raw_values = []
        for col in self.shared_features:
            if pd.api.types.is_numeric_dtype(self.df_a[col]):
                d = wasserstein_distance(
                    self.df_a[col].dropna(),
                    self.df_b[col].dropna()
                )
            else:
                d = self.categorical_drift(col)
            raw_values.append(float(d))

        # normalize to 0–1
        max_val = max(raw_values) if max(raw_values) > 0 else 1
        values = [v / max_val for v in raw_values]

        # close loop
        values.append(values[0])

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)

        labels = self.shared_features + [self.shared_features[0]]
        angles = np.linspace(0, 2 * np.pi, len(labels))

        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.3)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)

        return {"status": "ok", "radar_values": values, "figure": fig}

    # --------------------------------------------------------------------
    # 4) DOMAIN OVERLAP HEATMAP
    # --------------------------------------------------------------------
    def domain_overlap(self, bins=20):

        def numeric_overlap(a, b):
            a = a.dropna()
            b = b.dropna()
            if len(a) == 0 or len(b) == 0:
                return 0.0
            ha, _ = np.histogram(a, bins=bins, density=True)
            hb, _ = np.histogram(b, bins=bins, density=True)
            return float(np.sum(np.minimum(ha, hb)))

        feats = self.shared_features
        matrix = pd.DataFrame(0.0, index=feats, columns=feats)

        for c1 in feats:
            for c2 in feats:

                if pd.api.types.is_numeric_dtype(self.df_a[c1]) and \
                   pd.api.types.is_numeric_dtype(self.df_b[c2]):
                    val = numeric_overlap(self.df_a[c1], self.df_b[c2])
                else:
                    if c1 == c2:
                        val = self.categorical_overlap(c1)
                    else:
                        val = 0.0

                matrix.loc[c1, c2] = val

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix.values)
        fig.colorbar(im)
        ax.set_xticks(range(len(feats)))
        ax.set_yticks(range(len(feats)))
        ax.set_xticklabels(feats)
        ax.set_yticklabels(feats)
        ax.set_title("Domain Overlap Matrix")

        return {"status": "ok", "overlap_matrix": matrix, "figure": fig}
