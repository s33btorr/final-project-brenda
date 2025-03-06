from final_project_btb.config import (
    BLD,
)
from final_project_btb.data_management.cleaning_data import clean_data_extracted


def task_generate_csv(
    data=BLD / "page_for_detection.png", produces=BLD / "clean_table.csv"
):
    """Extract the data, arrange rows and columns, clean the dataframe.

    Args:
        data (str or Path): Path of image for detection and data extraction.
        produces (str or Path): Path where the cleaned CSV file will be saved.
    """
    df = clean_data_extracted(data)
    df.to_csv(produces)
