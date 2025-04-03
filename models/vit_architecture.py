import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from vit import ViT

# ======== Setup ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======== Transforms and Dataset ========
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Adjust to where your script is running (from 'models/' directory)
data_path = os.path.join("..", "data", "raw")
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
num_classes = len(full_dataset.classes)

# Split into 80% train / 20% test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ======== Instantiate Vision Transformer ========
model = ViT(
    image_size=128,
    patch_size=16,
    num_classes=num_classes,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

# ======== Loss and Optimizer ========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# ======== Training ========
def train(model, loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

# ======== Evaluation ========
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true += labels.cpu().tolist()
            y_pred += predicted.cpu().tolist()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return y_true, y_pred

# ======== Confusion Matrix ========
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Vision Transformer Confusion Matrix")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# ======== Run ========
train(model, train_loader, criterion, optimizer, epochs=10)
y_true, y_pred = evaluate(model, test_loader)
plot_confusion_matrix(y_true, y_pred, full_dataset.classes)
