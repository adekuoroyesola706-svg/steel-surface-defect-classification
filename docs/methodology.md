# Methodology

## Dataset
NEU-DET: 1,800 grayscale steel surface images across six defect classes.

## Split
The notebook recombines the available images and creates a stratified 70/15/15 train/validation/test split.

## Preprocessing
Images are resized to 224×224 RGB. Training augmentation includes rotation, shifts, zoom, brightness variation, and horizontal/vertical flips. Validation and test data are not augmented.

## Models
- Custom CNN trained from scratch.
- MobileNetV2 with ImageNet transfer learning and staged fine-tuning.
- ResNet50 with ImageNet transfer learning and staged fine-tuning.

## Explainability
Grad-CAM and SHAP are used to inspect model decisions.

## Evaluation
Accuracy, macro precision, macro recall, macro F1, weighted F1, AUC, confusion matrices, parameter count, model size, and inference time are considered.

## Reproducibility note
Final published metrics must come from the corrected model-to-test-generator evaluation pipeline. The original notebook contained a later evaluation cell that referenced a generic `model` variable and produced results inconsistent with the main comparison table.
