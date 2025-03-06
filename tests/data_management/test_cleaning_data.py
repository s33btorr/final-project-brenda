import pandas as pd
import pytest

from final_project_btb.config import BLD
from final_project_btb.data_management.cleaning_data import (
    _clean_number_columns,
    _concatenate_extra_rows,
    _delete_empty_rows,
    _delete_rows_with_na,
    clean_data_extracted,
)

data = BLD / "page_for_detection.png"


def test_clean_data_extracted_image_for_prediction_does_not_exists():
    with pytest.raises(FileNotFoundError):
        clean_data_extracted("not_existing_image.png")


def test_clean_data_extracted_df_correctly_obtained():
    df = clean_data_extracted(data)

    assert isinstance(df, pd.DataFrame), "Function is not returning a DataFrame"

    assert not df.empty, "DataFrame is empty"

    assert df.shape[0] > 0, "DataFrame has no rows"
    assert df.shape[1] > 0, "DataFrame has no columns"

    numeric_columns = [
        "paid_up_capital",
        "surp_and_prof",
        "deposits",
        "loans_and_discounts_stocks_and_securities",
        "cash_and_exchanges",
    ]
    for col in numeric_columns:
        assert pd.api.types.is_numeric_dtype(
            df[col]
        ), f"The column {col} does not contains only numbers."


def test_concatenate_extra_rows_is_correcting_unifying_rows():
    data = {
        "Col_1": ["Town1", "", "Town2"],
        "Col_2": ["Data1", "Data2", "Data3"],
    }
    df = pd.DataFrame(data)

    result_df = _concatenate_extra_rows(df)

    assert result_df.iloc[0, 0] == "Town1", "Rows are not unifying correctly."
    assert result_df.iloc[0, 1] == "Data1 Data2", "Rows are not unifying correctly."
    assert result_df.iloc[1, 0] != "Town2", "Rows were substituted instead of unified."


def test_delete_empty_rows_is_correctly_deleting_rows():
    data = {
        "Col_1": ["Town1", "", "Town2"],
        "Col_2": ["Data1", "", "Data3"],
    }
    df = pd.DataFrame(data)

    result_df = _delete_empty_rows(df)

    assert result_df.shape[0] == 2, "Empty row was not deleted correctly."
    assert result_df.iloc[1, 0] == "Town2", "Empty row was not deleted correctly."


def test_delete_rows_with_na_check_correctly_deleting():
    data = {
        "Col_1": [1, 2, 3],
        "Col_2": [4, None, 6],
        "Col_3": [None, None, None],
    }
    df = pd.DataFrame(data)

    result_df = _delete_rows_with_na(df, ["Col_2", "Col_3"])

    assert result_df.shape[0] == 2, "No correct elimination of rows with NaN."

    assert result_df.iloc[0, 0] == 1, "No correct elimination of rows with NaN."
    assert result_df.iloc[1, 0] == 3, "No correct elimination of rows with NaN."

    assert list(result_df.columns) == list(df.columns), "Columns were changed by error."


def test_clean_number_columns_enters_a_string_return_a_int():
    df_invalid = pd.DataFrame({"Col_1": [123, 456, 789]})
    with pytest.raises(TypeError):
        _clean_number_columns(df_invalid, ["Col_1"])

    df_valid = pd.DataFrame({"Col_1": ["123", "45a", "78.9", "abc", "0"]})
    result_df = _clean_number_columns(df_valid, ["Col_1"])

    assert isinstance(result_df, pd.DataFrame), "Function is not returning a DataFrame"
    assert pd.api.types.is_numeric_dtype(result_df["Col_1"]), "Col_1 is not numeric"
