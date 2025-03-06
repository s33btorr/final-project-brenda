import matplotlib.pyplot as plt
import pytask
from pdf2image import convert_from_path

from final_project_btb.analysis.data_extraction import extracting_data
from final_project_btb.analysis.prediction_pdf import generate_visualizer
from final_project_btb.config import (
    BLD,
    SRC,
)


def task_convert_pdf_to_image_and_save(
    data=SRC / "data" / "Data_banks.pdf", produces=BLD / "page_for_detection.png"
):
    """Extracts the first page from a PDF and saves it as an image."""
    pages = convert_from_path(data)
    pages[0].save(produces, "PNG")


def task_save_image_with_columns_predictions(
    data=BLD / "page_for_detection.png", produces=BLD / "predictions_columns.png"
):
    """Predicts the columns and saves an image with these predictions."""
    vision = generate_visualizer(data)
    plt.figure(figsize=(16, 24))
    plt.imshow(vision.get_image()[:, :, ::-1])
    plt.savefig(produces)


@pytask.mark.skip()
def task_generate_dataframe(
    data=BLD / "page_for_detection.png", produces=BLD / "table_no_cleaning.cvs"
):
    df = extracting_data(data)
    df.to_csv(produces)
