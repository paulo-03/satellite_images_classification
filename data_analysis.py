"""
This python script is used to gain insight of the data by making sure that all images
are the same size and displaying few images to understand and appreciate their content.

Author: Paulo Ribeiro
"""

import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter


# Function to load and get dimensions of all images in each category
def analyze_data(data_path: str, categories: list[str]):
    label_sizes = {}
    for category in categories:
        category_path = os.path.join(data_path, category)
        dimensions = []
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)

            if img is not None:
                dimensions.append(img.shape[:2])  # Record height and width
            else:
                err_msg = f"One image cannot be load ({img_name})"
                raise ValueError(err_msg)

        label_sizes[category] = dimensions

    # Print size statistics per label if not the same
    for label, sizes in label_sizes.items():
        size_counts = Counter(sizes)
        print(f"\nLabel '{label}' has {len(sizes)} images with the following size distribution:")
        for size, count in size_counts.items():
            print(f" - Size {size}: {count} images")


# Function to load and display a sample of 4 images per label
def display_sample_images(data_path: str, categories: list[str], display_size: tuple[int]):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 13))

    for i, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        image_files = os.listdir(category_path)[:4]  # Select the first 4 images

        for j, img_name in enumerate(image_files):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, display_size)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                axes[j, i].imshow(img_rgb)
                axes[j, i].text(0.5, -0.15, img_name, ha='center', va='top', transform=axes[j, i].transAxes, fontsize=8)
                # axes[j, i].axis('off')

        # Set label for the top of each column
        axes[0, i].set_title(category, fontsize=14)

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
