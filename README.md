# Steel Surface Defect Classification

Deep learning pipeline for six-class steel surface defect classification using a custom CNN, MobileNetV2, and ResNet50 with transfer learning, Grad-CAM, SHAP, and a Gradio inference interface.

## Project Overview

This project investigates automated classification of steel surface defects using the NEU-DET dataset.

The workflow covers:

1. Dataset inspection and stratified 70/15/15 splitting
2. Image preprocessing and augmentation
3. Custom CNN baseline
4. MobileNetV2 transfer learning and fine-tuning
5. ResNet50 transfer learning and fine-tuning
6. Quantitative model comparison
7. Grad-CAM and SHAP explainability
8. Error analysis
9. Interactive Gradio inference

## Defect Classes

- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

## Repository Structure

```text
notebooks/   Experiment and reproducibility notebook
src/         Reusable Python implementation
app/         Gradio demo
results/     Final figures and tables
models/      Model artifact instructions
docs/        Methodology
examples/    Optional inference examples
```

## Dataset

The project uses the NEU-DET steel surface defect dataset containing 1,800 grayscale images across six classes.

The dataset is **not included in this repository**. Place an extracted copy under `data/NEU-DET/` or set the `STEEL_DEFECT_DATA` environment variable to its location.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebook

Open:

`notebooks/steel_defect_classification_cleaned.ipynb`

The notebook has been cleaned to remove Colab/Google Drive-specific paths and the contradictory standalone evaluation/demo/export cells.

## Inference

After a validated model has been trained and saved:

```bash
python -m src.inference   --model models/resnet50_final.keras   --image examples/your_image.jpg
```

## Explainability

The project includes:

- Grad-CAM visual explanations
- SHAP feature-attribution explanations
- Misclassification/error analysis

## Results

Final numerical results should be generated from the corrected evaluation pipeline before being published here.

The original notebook contains a main comparison table reporting 98.89% Custom CNN accuracy, 98.15% MobileNetV2 accuracy, and 100% ResNet50 accuracy, but a later evaluation cell produces contradictory output because it references a generic `model` variable. Therefore those figures are **not treated as final published results until the corrected evaluation is rerun**.

## Important Reproducibility Note

Each model must be evaluated with its matching test generator:

```python
evaluate_model(custom_cnn, test_gen_cnn, "custom_cnn")
evaluate_model(mobilenet_model, test_gen, "mobilenetv2")
evaluate_model(resnet_model, test_gen_res, "resnet50")
```

This prevents accidentally evaluating one model against another model's generator.

## Technologies

Python · TensorFlow · Keras · scikit-learn · NumPy · pandas · Matplotlib · Seaborn · SHAP · Gradio

## Limitations

The benchmark is based on a relatively small dataset and a controlled test split. High benchmark performance should not automatically be interpreted as production-level generalization.

## Future Work

Potential extensions include:

- evaluation on larger or factory-specific datasets
- edge deployment with TensorFlow Lite
- model compression through quantization/pruning
- defect localization
- production integration

## License

Add an appropriate license before public release.
