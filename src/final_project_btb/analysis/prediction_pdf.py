import os

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

from final_project_btb.config import (
    model_trained_path,
    number_of_classes,
    number_of_detections_per_image,
    threshold,
)


def generate_visualizer(path_de_imagen):
    """Generates visualizations of object using a pre-trained model.

    Args:
        path_de_imagen (str): Path to the input image on which object detection
        predictions will be performed.

    Returns:
        Visualizer(detectron2.utils.visualizer.VisImage):
        An object that will be use checking the predictions made.
    """
    _fail_if_image_does_not_exists(path_de_imagen)
    image = cv2.imread(path_de_imagen)
    config_var = _set_up_predictor(
        model_trained_path, number_of_classes, number_of_detections_per_image, threshold
    )
    predictor = DefaultPredictor(config_var)
    outputs = predictor(image)

    vis = Visualizer(
        image[:, :, ::-1],
        metadata=MetadataCatalog.get("TestDataset_train"),
        scale=0.8,
        instance_mode=ColorMode.IMAGE_BW,
    )
    vis = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    return vis


def _set_up_predictor(
    model_trained_path, numero_de_clases, detecciones_por_imagen, grado_de_confianza
):
    """Sets up predictor parameters."""
    setup_logger()

    cfg = get_cfg()

    _fail_if_model_path_does_not_exist(model_trained_path)
    _fail_if_number_of_classes_is_not_integer(numero_de_clases)
    _fail_if_detections_per_image_is_not_integer(detecciones_por_imagen)
    _fail_if_threshold_is_not_valid(grado_de_confianza)

    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = numero_de_clases
    cfg.MODEL.WEIGHTS = str(model_trained_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = grado_de_confianza
    cfg.TEST.DETECTIONS_PER_IMAGE = detecciones_por_imagen
    cfg.MODEL.DEVICE = "cpu"  # not necessary if you have a GPU (also faster with gpu)

    return cfg


### Fail Functions


def _fail_if_image_does_not_exists(ruta):
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"This document {ruta} does not exists.")


def _fail_if_model_path_does_not_exist(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"This document {ruta} does not exists.")


def _fail_if_number_of_classes_is_not_integer(valor):
    if not isinstance(valor, int):
        raise TypeError(
            f"Number of classes should be an integer, and not {type(valor)}."
        )


def _fail_if_detections_per_image_is_not_integer(valor):
    if not isinstance(valor, int):
        raise TypeError(
            f"Number of detections should be an integer, and not {type(valor)}."
        )


def _fail_if_threshold_is_not_valid(valor):
    if not isinstance(valor, (int, float)):
        raise TypeError("threshold should be a number between 0 and 1.")
    if not 0 <= valor <= 1:
        raise ValueError("Threshold should be a number between 0 and 1.")
