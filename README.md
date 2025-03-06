# Detecting and Cleaning Table Data from PDFs Using Deep Learning and PyTesseract

![MIT license](https://img.shields.io/github/license/yourusername/your-repository)
[![image](https://zenodo.org/badge/12345678.svg)](https://zenodo.org/badge/latestdoi/12345678)
[![Documentation Status](https://readthedocs.org/projects/your-project-name/badge/?version=stable)](https://your-project-name.readthedocs.io/en/stable/)
[![image](https://github.com/yourusername/your-repository/actions/workflows/main.yml/badge.svg)](https://github.com/yourusername/your-repository/actions/workflows/main.yml)
[![image](https://codecov.io/gh/yourusername/your-repository/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/your-repository)

This project focuses on:

1. Training a deep learning model to detect tabular data on PDFs.
1. Detection and extraction of a specific PDF file with complex tables.
1. Cleaning of the data extracted.

It uses the model called `"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"` from
`Detectron2` for training the model. The database used in the training was made by me,
it contains 25 images in which the columns have been marked. For the detection of the
text `PyTesseract` was used.

> [!NOTE]
> **PyTesseract** can sometimes have issues reading words correctly. Some issues with
> OCR accuracy may require manual verification before serious use.

## How to Run the Project

To run this project, follow these steps:

### 1. Clone this repository

### 2. Create and activate environment

```console
$ mamba env create -f environment.yml
$ conda activate final_project_btb
```

### 3. Download data

You have two options to download the data:

1. Via Google Drive: Click on this link
   (\[https://drive.google.com/file/d/1ha7JIu2NRsnpCufi6PjMHNMyqfmNB_z8/view?usp=sharing\])
   and then click "Download".
1. Via Dropbox: Click on this link
   (\[https://www.dropbox.com/s/j3k3kkl97sw9ocy/model_final-2.pth?st=3nt83ul7&dl=0\])
   and then click "Download".

### 4. Place data in the data folder of src/final_project_btb

Path: final-project-s33btorr/src/final_project_btb/data

### 5. Run Pytask command

```console
pytask
```

## Short explanation of the project

### Motivation

My motivation for this project stems from the fact that I could not find any pre-trained
model, software, or package that could accurately read the table I needed given its
complexity. Therefore, I trained a model using images similar to those I need to
extract, allowing me to automate the extraction of a large number of pages in the
future. With other programs, this process would take hours and result in a significant
number of errors.

### Overview

In this project, I have trained a deep learning model to detect the columns of a table
from scanned PDFs using the **Roboflow** dataset I generated. After training, the model
can identify the positions of different table columns. The extracted data is then
processed and cleaned for analysis.

### Dataset

You can access the dataset used for training via the following link:
[Roboflow Dataset](https://app.roboflow.com/test-ypjyd/my-first-project-jqmvu/10)

### Training the Model

To train the model, I used the following approach:

1. The model was trained using a GPU provided by **Google Colab**.
1. The model was saved after training as `model_final-2.pth`.
1. The model is capable of detecting the columns in the table from a specific scanned
   PDF.

You can view and download to modify the code used to train the model in this notebook:
[Training Model Notebook](https://www.dropbox.com/s/dcgerv5i1yp217a/training_model.ipynb?st=5qaeufd1&dl=0)

### Making Predictions

Once the model is trained, it is saved as `model_final-2.pth`. This file is used to:

1. Extract the text using **PyTesseract**. I noticed that PyTesseract leaves a blank
   cell whenever the text goes to a new line. This can be used to determine the
   boundaries of each row in the table.
1. Predict column positions in new PDF tables with similar structure. Use this
   predictions to know where the text is located regarding the columns.

### Cleaning the Data

After extracting the data, the following cleaning steps were performed:

1. **Handling Missing Values**: If the first column is empty, it means that row is part
   of the previous one. These rows are merged accordingly.
1. **Numeric Fields**: Cleaned numeric fields to ensure they are in a format suitable
   for analysis (e.g., removing non-numeric characters).
1. **CVS and Graph Generation**: Generated the cvs file and some basic graphs to
   visualize the cleaned data.

## References

### Code for training model:

**Shen, Zejiang, Zhang, Kaixuan, & Dell, Melissa. (2020).** "A Large Dataset of
Historical Japanese Documents with Complex Layouts"
[arXiv:2004.08686](https://arxiv.org/abs/2004.08686)

### Model used for training:

Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., & Girshick, R. (2019). **Detectron2**.
Retrieved from
[https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2).
