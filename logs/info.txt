LOGS - Tensorboard Logs Directory Structure

Overview:
The logs directory is specifically designed to store TensorBoard logs generated during the training process. These logs can be used to visually track and analyze the performance and metrics of your models using TensorBoard.

Directory Structure:
logs/
│
├── acl/                     # Logs for ACL tear classification.
│   ├── axial/               # Logs for axial plane.
│   ├── coronal/             # Logs for coronal plane.
│   └── sagittal/            # Logs for sagittal plane.
│
└── *other_cases/            # Similar structure for other cases.

Usage:
Viewing Logs with TensorBoard: To visualize the training and validation metrics, launch TensorBoard and point it to the respective log directory. For example, if you wish to view the logs for ACL tear classification in the axial plane, use tensorboard --logdir=logs/acl/axial/.

Log Organization: Ensure that when generating logs during training, they are saved in the corresponding case and plane sub-directory for easy accessibility and organization.
