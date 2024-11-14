# Satellite Image Classification

This repository contains my implementations for a satellite image classification task provided by Swisscom as part of a coding test for my Master's thesis application in industry, completing my Master’s degree in Data Science. The project includes a baseline model using random forest with feature engineering, and a deep learning model using fine-tuning of ResNet18.

**Author:** [Paulo Ribeiro](mailto:paulo.ribeirodecarvalho@hotmail.com)

## Project Instructions

- **Dataset**: [Satellite Images](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)
- **Main Task**: Build two models to classify images into four categories:
  - A simple baseline model
  - A more complex method (e.g., deep learning), with explanations/interpretations of predictions

- **Challenge**: Are there pairs of images that are particularly challenging to distinguish?

## My Solution and Observations

1. **Model Comparison**: The deep learning model (ResNet18) outperformed the baseline, improving all metrics (accuracy, precision, recall, and F1-score) by 2-3%. However, both models struggled with certain misclassifications:
   - **Water images** were frequently mistaken for green areas.
   - The **baseline model** was especially sensitive to desert images, often predicting them as cloudy.

2. **Further Fine-tuning**: Increasing the number of training epochs could further enhance the model, as the loss trend suggests it hasn’t yet reached a plateau. This may bring the model closer to a perfect classifier.

3. **Baseline Performance**: Despite its simplicity, the baseline model performed reasonably well and is efficient in terms of computational resources for both training and inference.

> **_Note:_** This exercise was allocated 8 hours; I estimate I spent approximately 7 hours completing it.
