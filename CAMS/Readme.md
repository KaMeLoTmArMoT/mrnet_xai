CAMS - Directory Structure
Welcome to the CAMS directory. This directory is organized to facilitate the use of the CAM visual explanations for different cases and views. Below is a guide to help you understand and navigate the structure.

Directory Structure:
CAMS/
│
└── axial/
    ├── xxxx/              # Case folders, where xxxx ranges from 0000 to 0120.
    │   ├── cams/          # Directory containing heatmaps for the specific case.
    │   │   └── x.png      # Heatmap images, where x represents the image number.
    │   │
    │   └── slices/        # Directory containing original MRI slices for the specific case.
    │       └── x.png      # MRI slice images, where x represents the image number.
    │
    ├── *other_cases/      # Similar structure for other cases.
    │
    └── info.txt           # This info file.

Details:
Case Folders: Each case is represented by a unique folder named using a four-digit format (e.g., 0001, 0002, ... up to 0120).

CAM Heatmaps (cams/): Inside each case folder, the cams/ directory contains the CAM visual explanations (heatmaps) for the axial slices. Each heatmap image is named using a simple number format (x.png), where x is the slice number.

MRI Slices (slices/): Alongside the heatmaps, each case folder contains a slices/ directory. This directory holds the original MRI slices corresponding to the heatmaps. Again, each slice is named using a simple number format (x.png), mirroring the heatmap naming convention.
