import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SkinDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layer(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
num_classes = len(train_dataset.classes)
model = SkinDiseaseCNN(num_classes=num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
