"""
This python script implement the class used to fine-tune the ResNET18 model to enhance the 4
categories classification process.

Author: Paulo Ribeiro
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42


class ResNET():
    def __init__(self, data_dir: str, img_size: tuple[int, int], test_size: float, batch_size: int, epochs: int):
        self.data_dir = data_dir
        self.img_size = img_size  # (128, 128)
        self.test_size = test_size  # 0.2
        self.batch_size = batch_size
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_losses = None
        self.val_losses = None
        self.val_f1_scores = None

    def data_loader(self, show: bool=False):
        # Data transformations for training, validation, and test sets
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # taken from literature
            ]),
            'val_test': transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # use of transfer learning
            ])
        }

        # Load dataset
        full_dataset = datasets.ImageFolder(self.data_dir, transform=data_transforms['train'])

        # Split into train, validation, and test sets without carrying so much of balanced label, need to finish the project
        train_size = int((1 - self.test_size) * len(full_dataset))
        val_size = int(0.15 * train_size)
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED)
        )

        # Apply separate transformations for validation and test sets
        val_dataset.dataset.transform = data_transforms['val_test']
        test_dataset.dataset.transform = data_transforms['val_test']

        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Class labels
        self.class_names = full_dataset.classes
        print("Class names:", self.class_names)

        if show:
            self._show_samples()

    def _show_samples(self):
        # Get one batch of 10 images from each loader
        loaders = [("Train", self.train_loader), ("Validation", self.val_loader), ("Test", self.test_loader)]
        fig, axes = plt.subplots(3, 10, figsize=(20, 6))

        for i, (name, loader) in enumerate(loaders):
            images, labels = next(iter(loader))

            for j in range(10):  # first 10 images in each loader
                img = images[j].numpy().transpose((1, 2, 0))  # move channels to last dimension
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize
                img = img.clip(0, 1)  # Clip to valid range

                axes[i, j].imshow(img)
                axes[i, j].axis("off")
                label_idx = labels[j].item()
                axes[i, j].set_title(self.class_names[label_idx], fontsize=8)

        # Set row titles
        for i, (name, _) in enumerate(loaders):
            axes[i, 0].set_ylabel(name, fontsize=12, rotation=0, labelpad=50, weight='bold')

        plt.tight_layout()
        plt.show()

    def load_model(self):
            # Load the pre-trained ResNet18 model
            model = models.resnet18(pretrained=True)

            # Freeze all layers except the final fully connected layer
            for param in model.parameters():
                param.requires_grad = False

            # Modify the final layer to match the number of classes (4 in this case)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(self.class_names))

            # Move the model to the appropriate device
            self.model = model.to(self.device)

            # Loss and optimizer
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only training the final layer

    def fine_tune(self):
        train_losses, val_losses = [], []
        val_f1_scores = []

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # Compute F1 score
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            val_f1_scores.append(val_f1)

            # Record average losses for this epoch
            train_losses.append(running_loss / len(self.train_loader))
            val_losses.append(val_loss / len(self.val_loader))

            print(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {train_losses[-1]:.4f}, "
                  f"Val Loss: {val_losses[-1]:.4f}, Val F1 Score: {val_f1:.4f}")

        self.train_losses, self.val_losses, self.val_f1_scores = train_losses, val_losses, val_f1_scores

    def show_training(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 4))

        # Plot Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Loss Evolution")

        # Plot F1 Score
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.val_f1_scores, label='Val F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title("F1 Score Evolution")

        plt.show()

    def evaluate(self):
        self.model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # Generate and print the classification report
        report = classification_report(test_labels, test_preds, target_names=self.class_names)
        print("Classification Report:\n", report)

        # Generate the confusion matrix
        cm = confusion_matrix(test_labels, test_preds)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
