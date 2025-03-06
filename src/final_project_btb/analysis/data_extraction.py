import os

import cv2
import pandas as pd
import pytesseract
from detectron2.engine import DefaultPredictor

from final_project_btb.analysis.prediction_pdf import _set_up_predictor
from final_project_btb.config import (
    model_trained_path,
    number_of_classes,
    number_of_detections_per_image,
    threshold,
)


def extracting_data(image_path):
    """Extracts tabular data from an image using predictions.

    Args:
        image_path (str): Path to the image to extract data from.

    Returns:
        df(pd.DataFrame): A DataFrame containing the extracted table data.
    """
    _fail_if_image_does_not_exists(image_path)
    image = cv2.imread(image_path)

    config_var = _set_up_predictor(
        model_trained_path, number_of_classes, number_of_detections_per_image, threshold
    )
    predictor = DefaultPredictor(config_var)
    output = predictor(image)

    columns_predictions = _saving_columns_predictions(output)
    results_of_ocr = _reading_ocr_text(image)
    data_of_table = _extracting_cells(columns_predictions, results_of_ocr)

    df = pd.DataFrame(
        data_of_table, columns=[f"Col_{i+1}" for i in range(len(columns_predictions))]
    )

    return df


def _reading_ocr_text(path_de_imagen):
    """Extracts OCR (optical character recognition) text from an image."""
    ocr_results = pytesseract.image_to_data(
        path_de_imagen, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    return ocr_results


def _saving_columns_predictions(predictions):
    """Extracts and sorts column bounding boxes from the model's predictions."""
    boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
    boxes = sorted(boxes, key=lambda x: x[0])
    return boxes


def _extracting_cells(boxes, ocr_results):
    """Extracts the table's cell data based on column boxes and OCR results.

    Args:
        boxes (list): List of predicted column bounding boxes.
        ocr_results (dict): OCR results containing the extracted text and positions.

    Returns:
        tabledata(list): List of rows, containing cell data based on the OCR results.
    """
    table_data = []
    current_row = [""] * len(boxes)

    for i in range(len(ocr_results["text"])):
        word = ocr_results["text"][i].strip()
        x, y, w, h = (
            ocr_results["left"][i],
            ocr_results["top"][i],
            ocr_results["width"][i],
            ocr_results["height"][i],
        )

        if not word:
            table_data.append(current_row)
            current_row = [""] * len(boxes)

        for col_idx, (x1, y1, x2, y2) in enumerate(boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                current_row[col_idx] += (" " + word) if current_row[col_idx] else word

    if any(current_row):
        table_data.append(current_row)

    return table_data


# Fail functions


def _fail_if_image_does_not_exists(ruta):
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"This document {ruta} does not exists.")
