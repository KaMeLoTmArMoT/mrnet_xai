SHAPS - Directory Structure
Welcome to the SHAPS directory. This directory is organized to facilitate the use of the CAM visual explanations for different cases and views. Below is a guide to help you understand and navigate the structure.

Directory Structure:
SHAPS/
│
└── axial/
    ├── xxxx/              # Case folders, where xxxx ranges from 0 to x.
    │   └── x.png          # Heatmap images, where x represents the image number.
    │
    ├── *other_cases/      # Similar structure for other cases.
    │
    └── info.txt           # This info file.

Details:
Case Folders: Each case is represented by a unique folder named using a four-digit format (e.g., 0001, 0002, ... up to 0120).

SHAP Heatmaps (shaps/): Inside each case folder, the shaps/ directory contains the CAM visual explanations (heatmaps) for the axial slices. Each heatmap image is named using a simple number format (x.png), where x is the slice number.
