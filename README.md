# xai_mrnet

Explainable Artificial Intelligence (XAI) applied to the MRNet dataset.

## Description

This repository provides tools and methods to visualize and understand the decisions made by deep learning models
trained on the MRNet dataset. We employ techniques like Class Activation Mapping (CAM) to highlight regions in MRI
slices that significantly influence the model's prediction, aiding in model interpretability.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [VIS](#vis)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/xai_mrnet.git
    cd xai_mrnet
    ```

2. Install the required packages

    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Ensure you have the MRNet dataset available in the data/ directory.
2. Run the main script to generate Class Activation Maps (CAM) for selected MRI slices:

## Datasets

The datasets used in this project are:

### MRNet
- **Description**: The MRNet dataset consists of knee MRI scans collected from Stanford University Medical Center. The dataset contains 1,370 knee MRI exams performed at Stanford University Medical Center.
- **Link**: [MRNet Competition](https://stanfordmlgroup.github.io/competitions/mrnet/)
- **Citation**:

### Knee MRI
- **Description**: This is a database of knee MRI scans and was part of a competition to stimulate research in automated techniques for the interpretation of 3D MRI scans.
- **Link**: [Knee MRI Dataset](http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/)
- **Citation**:

## Vis
The "Vis" program is designed to visualize MRI images of patients. It can render CAMs (Class Activation Maps), raw MRI slices, and SHAP (SHapley Additive exPlanations) images. Each patient's data is labeled and presented in an organized interface that displays the axial, coronal, and sagittal planes. Additionally, the program provides a prediction label and compares it with the actual ground truth to indicate the accuracy of the prediction in the form of "True Positive", "True Negative", "False Positive", and "False Negative".

### Key Bindings:
TODO

### Usage

Navigating the Interface: Once the program starts, you will see the MRI images displayed. Use the arrow keys to navigate through different patients or change visualization types.

Understanding the Display: The top section of the interface displays information about the patient, the ground truth label, the prediction label, and its status (e.g., True Negative). Below that, you will see the MRI images for the axial, coronal, and sagittal planes. The left sidebar indicates the visualization type currently being viewed.

Exiting: To exit the program, simply press the Esc key.

# Examples
![images\TP.png](images\TP.png)
![images\TN.png](images\TN.png)
![(images\FP.png](images\FP.png)
![images\FN.png](images\FN.png)
