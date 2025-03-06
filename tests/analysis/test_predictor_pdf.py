import pytest

from final_project_btb.analysis.prediction_pdf import (
    _set_up_predictor,
    generate_visualizer,
)
from final_project_btb.config import (
    model_trained_path,
    number_of_classes,
    number_of_detections_per_image,
    threshold,
)


def test_generate_visualizer_image_for_prediction_does_not_exists():
    with pytest.raises(FileNotFoundError):
        generate_visualizer("not_existing_image.png")


def test__set_up_predictor_verify_that_model_trained_path_exists():
    with pytest.raises(FileNotFoundError):
        _set_up_predictor(
            "path_to_non_existent_model.pth",
            number_of_classes,
            number_of_detections_per_image,
            threshold,
        )


@pytest.mark.parametrize("invalid_input", ["a", 8.3, -1.5])
def test__set_up_predictor_number_of_classes_is_integer(invalid_input):
    with pytest.raises(TypeError):
        _set_up_predictor(
            model_trained_path, invalid_input, number_of_detections_per_image, threshold
        )


@pytest.mark.parametrize("invalid_input", ["a", 8.3, -1.5])
def test__set_up_predictor_detections_per_image_is_integer(invalid_input):
    with pytest.raises(TypeError):
        _set_up_predictor(
            model_trained_path, number_of_classes, invalid_input, threshold
        )


@pytest.mark.parametrize("invalid_input", ["a", 8.3, -1.5, 2])
def test__set_up_predictor_threshold_is_between_zero_and_one(invalid_input):
    with pytest.raises((TypeError, ValueError)):
        _set_up_predictor(
            model_trained_path,
            number_of_classes,
            number_of_detections_per_image,
            invalid_input,
        )
