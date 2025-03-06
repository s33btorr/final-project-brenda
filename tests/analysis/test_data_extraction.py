import pandas as pd
import pytest

from final_project_btb.analysis.data_extraction import (
    extracting_data,
)
from final_project_btb.config import BLD

data = BLD / "page_for_detection.png"


def test_extracting_data_image_for_prediction_does_not_exists():
    with pytest.raises(FileNotFoundError):
        extracting_data("not_existing_image.png")


def test_extracting_data_valid_image():
    df = extracting_data(data)

    assert isinstance(df, pd.DataFrame), "Function is not returning a DataFrame"

    assert not df.empty, "DataFrame is empty"

    assert df.shape[0] > 0, "DataFrame has no rows"
    assert df.shape[1] > 0, "DataFrame has no columns"
