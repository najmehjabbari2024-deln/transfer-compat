from typing import Dict, Any
import pandas as pd

class DatasetComparator:
    """
    Basic class for dataset compatibility analysis.
    """

    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame):
        self.df_a = df_a
        self.df_b = df_b

    def shape_comparison(self) -> Dict[str, Any]:
        """
        Compare the shapes of two datasets.
        """
        return {
            "dataset_a_rows": self.df_a.shape[0],
            "dataset_b_rows": self.df_b.shape[0],
            "dataset_a_columns": self.df_a.shape[1],
            "dataset_b_columns": self.df_b.shape[1],
            "same_number_of_columns": self.df_a.shape[1] == self.df_b.shape[1],
        }
 
