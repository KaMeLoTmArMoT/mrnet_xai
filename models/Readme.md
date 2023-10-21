MODELS - Models Directory Structure

Overview:
The models directory houses the trained model checkpoints. These checkpoints are generated during the training process and can be later utilized for inferencing, generating CAMs, SHAPs, and further evaluation.

Directory Structure:
models/
│
├── model_prefix_ad_{plane}_val_auc_{val_auc}_train_auc_{train_auc}_epoch_{epoch}.pt
│
└── *other_models/            # Jupyter Notebook checkpoints (optional and auto-generated).

Note: {plane}, {val_auc}, {train_auc}, and {epoch} are placeholders and will be replaced with the appropriate values during the model saving process.

Usage:
Model Checkpoints: The model checkpoint filenames are informative. They contain details about the plane (e.g., axial, coronal, sagittal), validation AUC, training AUC, and the epoch at which the checkpoint was saved. This naming convention aids in quick identification and selection of models for further tasks.

Loading Models: When you need to use a model for inference or any post-processing tasks, you can directly load the checkpoint from this directory using PyTorch's load functionality.
