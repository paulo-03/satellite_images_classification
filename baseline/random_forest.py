"""
This paython script implement the classifier class used to classify the satellite images.

Author: Paulo Ribeiro
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42


class Classifier:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=SEED)

    def train(self, train_features, train_labels):
        self.model.fit(train_features, train_labels)

    def evaluate(self, features: np.ndarray[float], labels: list[int], img_paths: list[str]):
        predictions = self.model.predict(features)
        report = classification_report(
            labels, predictions, target_names=["cloudy", "desert", "green_area", "water"]
        )
        print(report)

        # Return the list of all images that have been wrongly classified and the prediction
        error_ids = np.where(predictions != labels)[0]
        misclassified_images = np.asarray(img_paths)[error_ids]
        wrong_predictions = predictions[error_ids]

        # Generate the confusion matrix
        cm = confusion_matrix(labels, predictions)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["cloudy", "desert", "green_area", "water"],
                    yticklabels=["cloudy", "desert", "green_area", "water"])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
        return misclassified_images, wrong_predictions

    @staticmethod
    def look_at_misclassified(misclassified_imgs: np.ndarray[str], wrong_predictions: np.ndarray[int], idx: int):
        plt.figure(figsize=(6, 6))  # You can adjust the figure size as needed

        img_path = misclassified_imgs[idx]
        img = cv2.imread(img_path)
        img_name = img_path.split("/")[-1]
        label_to_class = {0: 'cloudy', 1: 'desert', 2: 'green_area', 3: 'water'}
        wrong_prediction = label_to_class[wrong_predictions[idx]]

        # Convert BGR to RGB for correct color display, because of cv2 loading
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img_rgb)  # Display the image
        plt.axis('off')  # Hide the axis

        # Add the image name below the image
        plt.text(0.5, -0.1, f"{img_name} predicted as {wrong_prediction}", ha='center', va='top', fontsize=10,
                 transform=plt.gca().transAxes)

        plt.show()
