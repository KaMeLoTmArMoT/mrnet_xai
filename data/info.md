DATA - Dataset Directory Structure

Welcome to the data directory. This directory is structured to facilitate the organization and accessibility of the MRNet dataset and its corresponding labels. Here's a guide to help you understand and navigate the structure.

Directory Structure:
data/
│
├── train/                   # Training data folder.
│   ├── axial/               # Axial MRI scans for training.
│   ├── coronal/             # Coronal MRI scans for training.
│   └── sagittal/            # Sagittal MRI scans for training.
│
├── valid/                   # Validation data folder.
│   ├── axial/               # Axial MRI scans for validation.
│   ├── coronal/             # Coronal MRI scans for validation.
│   └── sagittal/            # Sagittal MRI scans for validation.
│
├── results.csv              # generated Results file (used for VIS).
│
├── train-abnormal.csv       # Training labels for abnormality.
├── train-acl.csv            # Training labels for ACL tears.
├── train-meniscus.csv       # Training labels for meniscal tears.
│
├── valid-abnormal.csv       # Validation labels for abnormality.
├── valid-acl.csv            # Validation labels for ACL tears (used for VIS, starts from 0000).
├── valid-acl_shifted.csv    # Shifted validation labels for ACL tears.
└── valid-meniscus.csv       # Validation labels for meniscal tears.

Details:
MRI Scans: MRI scans are organized into train and valid directories for training and validation data, respectively. Each of these directories is further divided by the MRI scan plane: axial, coronal, and sagittal.

Labels: Label files are CSV files provided for both training and validation sets. These files contain the ground truth labels for different conditions:

Abnormality (*-abnormal.csv)
ACL tears (*-acl.csv)
Meniscal tears (*-meniscus.csv)
The prefix train- or valid- denotes whether the labels correspond to the training or validation dataset.

Special Note: The valid-acl_shifted.csv file contains shifted labels for ACL tears in the validation set. Ensure to use the correct label file based on your specific needs.

Usage:
Uploading Dataset:
Upload your MRI scans into the corresponding plane directory (axial, coronal, sagittal) under the train or valid directories, depending on whether they are training or validation scans.
Uploading Labels:
Place the provided CSV label files directly under the data directory.
