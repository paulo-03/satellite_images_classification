"""
Python script used to vectorize the satellite images, to later feed the classifier
to predict the class of the images.

Author: Paulo Ribeiro
"""

import os
import cv2
import numpy as np
import random
import pandas as pd
from collections import Counter
from IPython.core.display_functions import display
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# Gather image paths and labels
def load_split_image_paths(data_path: str, categories: list[str], test_size: float = 0.2):
    image_paths = []
    labels = []
    labels_to_class_map = {}
    for label, category in enumerate(categories):
        labels_to_class_map[category] = label
        category_path = os.path.join(data_path, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            image_paths.append(img_path)
            labels.append(label)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=SEED, stratify=labels
    )

    # Quick display to ensure the user that the data has been split accordingly to the initial class distribution
    label_distribution = [Counter(train_labels), Counter(test_labels)]

    # Create a DataFrame to display the label counts
    label_counts_df = pd.DataFrame(
        {
            0: [label_distribution[0].get(0, None), label_distribution[1].get(0, None)],
            1: [label_distribution[0].get(1, None), label_distribution[1].get(1, None)],
            2: [label_distribution[0].get(2, None), label_distribution[1].get(2, None)],
            3: [label_distribution[0].get(3, None), label_distribution[1].get(3, None)],
        },
        index=["train", "test"]
    )

    # Print the DataFrame
    print(label_counts_df)
    print(labels_to_class_map)

    return train_paths, test_paths, train_labels, test_labels


class ImageVectorization:
    def __init__(self):
        self.scaler = StandardScaler()  # To standardize each feature

    @staticmethod
    def _load_image(img_path):
        return cv2.imread(img_path)

    @staticmethod
    def _color_features(img):
        # Calculate mean and std of RGB channels
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))
        return np.concatenate([mean, std])

    @staticmethod
    def _texture_features_glcm(img_gray):
        glcm = graycomatrix(img_gray, [1], [0, np.pi / 2])
        contrast = graycoprops(glcm, 'contrast').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        return np.array([contrast, correlation, energy, homogeneity])

    def vectorize_image(self, img_path):
        img = self._load_image(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract features
        color_feats = self._color_features(img)
        glcm_feats = self._texture_features_glcm(img_gray)

        # Concatenate all features into a single vector
        feature_vector = np.concatenate([color_feats, glcm_feats])
        return feature_vector

    def transform(self, image_paths, show_df: bool = False):
        features = [self.vectorize_image(img_path) for img_path in image_paths]  # Vectorize each image
        features = self.scaler.fit_transform(features)  # Standardize features

        if show_df:
            features_df = pd.DataFrame(
                features,
                columns=["red_mean", "green_mean", "blue_mean",
                         "red_std", "green_std", "blue_std",
                         "gray_contrast", "gray_correlation", "gray_energy", "gray_homogeneity"]
            )
            display(features_df.head())

        return np.array(features)
