#the code for this vision transformer was found on the following github
#https://github.com/miladfa7/Image-Classification-Vision-Transformer/blob/master/notebooks/ViT-Model-Brain-Tumor-Detection.ipynb

#the actual vision transformer model is the following "google/vit-base-patch16-224-in21k" from Google and Hugging Face

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from PIL import Image
import os
from collections import Counter

# Preprocessing and Dataset Setup
# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
train_data_path = '/content/interim'  
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
num_classes = len(train_dataset.classes)
print(f"Detected {num_classes} classes: {train_dataset.classes}")

# Labels to apply synthetic augmentation to
labels_to_transform = list(range(num_classes))

# Custom Dataset with optional synthetic augmentation
class CustomDataset(Dataset):
    def __init__(self, real_data, labels_to_generate, transform=None):
        self.real_data = real_data
        self.labels_to_generate = labels_to_generate
        self.transform = transform
        self.samples = self.real_data.samples
        self.loader = self.real_data.loader
        self.base_transform = self.real_data.transform
        self.class_to_idx = getattr(self.real_data, 'class_to_idx', {})
        self.classes = getattr(self.real_data, 'classes', [])

        self.class_counts = {}
        self.target_class_indices = {label: [] for label in self.labels_to_generate}
        for idx, (_, label) in enumerate(self.samples):
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
            if label in self.labels_to_generate:
                self.target_class_indices[label].append(idx)

        self.synthetic_samples_per_label = {}
        self.total_synthetic_samples = 0
        max_class_count = max(self.class_counts.values()) if self.class_counts else 0
        for label in self.labels_to_generate:
            if self.class_counts.get(label, 0) < max_class_count:
                samples_needed = 500
                self.synthetic_samples_per_label[label] = samples_needed
                self.total_synthetic_samples += samples_needed
            else:
                self.synthetic_samples_per_label[label] = 0

        self.synthetic_index_map = [label for label in self.labels_to_generate
                                    for _ in range(self.synthetic_samples_per_label.get(label, 0))]
        self.synthetic_cache = {}

    def __len__(self):
        return len(self.samples) + self.total_synthetic_samples

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            synthetic_idx = idx - len(self.samples)
            if synthetic_idx in self.synthetic_cache:
                return self.synthetic_cache[synthetic_idx]
            label = self.synthetic_index_map[synthetic_idx]
            sample = self._generate_synthetic_sample(label)
            self.synthetic_cache[synthetic_idx] = sample
            return sample

        path, label = self.samples[idx]
        image = self.loader(path)
        image = self.transform(image) if label in self.labels_to_generate else self.base_transform(image)
        return image, label

    def _generate_synthetic_sample(self, label):
        if not self.target_class_indices[label]:
            image = self.loader(self.samples[0][0])
        else:
            idx = random.choice(self.target_class_indices[label])
            image = self.loader(self.samples[idx][0])
        return self.transform(image), label

# Build augmented dataset
augmented_dataset = CustomDataset(train_dataset, labels_to_transform, transform=transform)

# Check label distribution
label_counter = Counter(label for _, label in augmented_dataset)
for label in labels_to_transform:
    label_counter[label] += augmented_dataset.synthetic_samples_per_label.get(label, 0)
print(f"Label distribution: {dict(label_counter)}")

plt.bar(label_counter.keys(), label_counter.values())
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.show()

# Split data
train_size = int(0.8 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size
train_dataset, test_dataset = random_split(augmented_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pre-trained ViT from Google and Hugging Face
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

# Need to override classifier head
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Set device (force CPU if debugging)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training & Evaluation

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_model(model, loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            if labels.max() >= num_classes or labels.min() < 0:
                print(f" Invalid label found in batch: {labels}")
                continue

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")

def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds += predicted.cpu().tolist()
            all_labels += labels.cpu().tolist()

    acc = 100 * correct / total
    print(f"\nTest Accuracy: {acc:.2f}%")
    return all_labels, all_preds

train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
y_true, y_pred = evaluate_model(model, test_loader, device)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_dataset.dataset.classes,
            yticklabels=train_dataset.dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("ViT Confusion Matrix")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()