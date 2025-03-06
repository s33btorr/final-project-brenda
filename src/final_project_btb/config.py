"""All the general configuration of the project."""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()
model_trained_path = SRC / "data" / "model_final-2.pth"

BLD = ROOT.joinpath("bld").resolve()


# Parameters for setting up the predictor
number_of_classes = 2
threshold = 0.5
number_of_detections_per_image = 200

# Columns that contains numbers
numeric_cols = [
    "paid_up_capital",
    "surp_and_prof",
    "deposits",
    "loans_and_discounts_stocks_and_securities",
    "cash_and_exchanges",
]
