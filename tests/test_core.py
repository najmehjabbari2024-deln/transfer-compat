import pandas as pd
from transfer_compat.core import DatasetComparator

def test_shape_comparison():
    df_a = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6]
    })

    df_b = pd.DataFrame({
        "x": [9, 8],
        "y": [7, 6]
    })

    comparator = DatasetComparator(df_a, df_b)
    result = comparator.shape_comparison()

    assert result["dataset_a_rows"] == 3
    assert result["dataset_b_rows"] == 2
    assert result["dataset_a_columns"] == 2
    assert result["dataset_b_columns"] == 2
    assert result["same_number_of_columns"] is True
