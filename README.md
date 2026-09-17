# Steel Surface Defect Classification Using Transfer Learning and Deep Convolutional Neural Networks

## Overview

This project investigates the application of deep learning and transfer learning for automated classification of surface defects in steel manufacturing.

The objective is to develop a computer-vision-based inspection system capable of automatically identifying different types of steel surface defects from images. Such systems can support automated quality inspection and reduce dependence on manual visual inspection in industrial manufacturing environments.

## Research Motivation

Surface defects can significantly affect the quality and reliability of manufactured steel products. Traditional visual inspection can be time-consuming and may be affected by human subjectivity.

This project explores whether deep convolutional neural networks and transfer learning can provide an effective approach for automated steel surface defect classification.

## Research Questions

* Can transfer learning provide effective representations for steel surface defect images?
* Which deep-learning architecture provides the best classification performance?
* How well can the trained models distinguish between different defect categories?
* Which preprocessing and augmentation techniques improve model performance?

## Methodology

The project follows the following workflow:

1. Dataset preparation
2. Exploratory data analysis
3. Image preprocessing
4. Data augmentation
5. Transfer-learning model selection
6. Model training and validation
7. Performance evaluation
8. Error and misclassification analysis

## Technologies

* Python
* PyTorch / TensorFlow
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Scikit-learn

## Models

The project evaluates transfer-learning-based convolutional neural networks.

Models evaluated:

* custom_cnn
* mobilenetv2
* resnet50

## Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Additional evaluation and error analysis are included in the `results/` directory.

## Results
Custom CNN: 98.89% accuracy with balanced precision/recall/F1 (~98.90%), showing strong performance even without pre-training.
MobileNetV2: 98.15% accuracy, best efficiency trade-off — fast (8.7ms inference, ~115 images/sec) and lightweight (25MB), ideal for edge/industrial deployment.
ResNet50: 100% accuracy (perfect classification), best overall performance via transfer learning and deep residual connections, though the small test set (270 images) means this result should be interpreted cautiously.

### Model Performance
Table 2 - Performance Metrics Comparison
Model        Accuracy     Macro Precision   Macro Recall  Macro F1
custom_cnn   0.9889           0.9890         0.9889        0.9889
mobilenetv2  0.9815           0.9828         0.9815        0.9815
resnet50     1.000            1.0000         1.0000        1.0000

### Confusion Matrix
Confusion matrices for each model are shown in Figures 7-9. They reveal the per-class performance and common misclassifications, providing insights into which defect types pose the greatest challenges for each approach. The diagonal dominance in all confusion matrices indicates strong classification performance, while off-diagonal elements reveal patterns of confusion between specific defect classes. 

### Sample Predictions
A web-based interface was developed using **Gradio** to demonstrate the practical application of the trained steel surface defect classification models. The interface allows users to upload steel surface images and receive **real-time defect predictions with confidence scores**. It supports common image formats such as JPG and PNG and provides an image preview, predicted defect class, and prediction confidence.

The system uses the fine-tuned **ResNet50 model**, which was the best-performing model in the experiments. Uploaded images are resized to **224×224 pixels**, normalized to the **[0,1] range**, and processed by the model to identify one of six defect types: **crazing, inclusion, patches, pitted surface, rolled-in scale, or scratches**. The highest-probability class and its confidence score are presented to the user.

The Gradio interface features a simple, user-friendly design with descriptive labels and example inputs, making it accessible to non-technical users. It runs locally and can be adapted for web deployment, demonstrating the potential for practical use of the developed steel defect classification system in industrial environments.


## Research Significance
The project demonstrates the potential of deep learning and transfer learning for automated visual quality inspection in steel manufacturing.

The work is particularly relevant to intelligent manufacturing, industrial computer vision, automated quality control, and AI-based inspection systems.

## Future Work

Potential extensions include:

* Object detection and defect localization
* Semantic or instance segmentation
* Few-shot and self-supervised learning
* Explainable AI for industrial inspection
* Domain adaptation across different steel products and imaging conditions
* Deployment of the trained model for real-time inspection

## Reproducibility

Installation instructions, dependencies, training procedures, and evaluation steps are provided in this repository.

The dataset is not included in the repository where redistribution is restricted. Instructions for obtaining and preparing the dataset are provided separately.

## Author

Adekuoroye Sola Emmanuel

Final-Year Undergraduate Researcher

Research interests: Artificial Intelligence, Machine Learning, Computer Vision, Intelligent Manufacturing, Industrial Inspection, Anomaly Detection, Predictive Maintenance

Research interests: Artificial Intelligence, Machine Learning, Computer Vision, Intelligent Manufacturing, Industrial Inspection, Anomaly Detection, Predictive Maintenance
