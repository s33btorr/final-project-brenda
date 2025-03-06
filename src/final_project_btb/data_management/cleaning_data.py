import os

import numpy as np
import pandas as pd

from final_project_btb.analysis.data_extraction import extracting_data
from final_project_btb.config import numeric_cols


def clean_data_extracted(image_path):
    """Extract the data, arrange rows and columns, clean the dataframe.

    Args:
        image_path (str): Path to the input image on which object detection
        predictions will be performed.

    Returns:
        df(DataFrame): Semi-Clean DataFrame ready to graph some results.
    """
    _fail_if_image_does_not_exists(image_path)
    original_df = extracting_data(image_path)
    original_df = _concatenate_extra_rows(original_df)
    original_df = _delete_empty_rows(original_df)

    rename_dict = {
        "Col_1": "town_and_county",
        "Col_2": "name_of_bank",
        "Col_3": "president",
        "Col_4": "vice_president",
        "Col_5": "cashier",
        "Col_6": "asst_cashier",
        "Col_7": "paid_up_capital",
        "Col_8": "surp_and_prof",
        "Col_9": "deposits",
        "Col_10": "loans_and_discounts_stocks_and_securities",
        "Col_11": "cash_and_exchanges",
        "Col_12": "principal_correspondence",
    }
    df = original_df.copy()
    df = df.rename(columns=rename_dict)

    df = _clean_number_columns(df, numeric_cols)
    df = _delete_rows_with_na(df, numeric_cols)

    return df


def _concatenate_extra_rows(base_de_datos):
    """Concatenates rows with empty cell in first column."""
    df_array = base_de_datos.to_numpy(copy=True)

    for i in range(1, len(df_array)):
        if df_array[i, 0] == "":
            df_array[i - 1] = np.where(
                df_array[i] != "", df_array[i - 1] + " " + df_array[i], df_array[i - 1]
            )
            df_array[i] = ""

    df_final = pd.DataFrame(df_array, columns=base_de_datos.columns)
    return df_final


def _delete_empty_rows(base_de_datos):
    """Eliminates every row with all empty cells."""
    df_without_empty_rows = base_de_datos.replace("", np.nan).dropna(how="all")
    df_without_empty_column1 = df_without_empty_rows.dropna(
        subset=[df_without_empty_rows.columns[0]]
    )

    return df_without_empty_column1.reset_index(drop=True)


def _clean_number_columns(df, name_columns):
    """Eliminates every character that is not a number in number columns."""
    df_clean = df.copy()
    _fail_if_values_are_not_strings(df, name_columns)
    for col in name_columns:
        df_clean[col] = df_clean[col].astype(str).replace(r"\D", "", regex=True)
        df_clean[col] = pd.to_numeric(df_clean[col]).astype(pd.UInt32Dtype())
    return df_clean


def _delete_rows_with_na(df, name_columns):
    """Eliminates every row with all na."""
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=name_columns, how="all")
    return df_clean


# Fail functions


def _fail_if_image_does_not_exists(ruta):
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"This document {ruta} does not exists.")


def _fail_if_values_are_not_strings(data, columns):
    for col in columns:
        if not all(data[col].apply(lambda x: isinstance(x, str) or pd.isna(x))):
            raise TypeError(f"Column {col} must contain only strings or NaN.")
