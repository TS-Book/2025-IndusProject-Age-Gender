# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:29:17 2025

@author: thana

No use of face detection
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2  # ใช้ Wide ResNet

# 📌 1. กำหนด Paths
dataset_path = r"D:\University\3\3_2\Indus based\AGE_Detection\Datasets"
save_model_path = r"D:\University\3\3_2\Indus based\AGE_Detection\Model\age_model.pth"

# 📌 2. Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize เป็น 224x224
    transforms.ToTensor(),          # แปลงเป็น Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize แบบ ImageNet
])

# 📌 3. Load Dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
train_size = int(0.8 * len(dataset))  # 80% Train, 20% Validation
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 📌 4. Create Dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 📌 5. Load WRN-16-8 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = wide_resnet50_2(pretrained=True)  # โหลด Wide ResNet ตัวใหญ่ขึ้นแทน WRN-16-8
model.fc = nn.Linear(model.fc.in_features, 5)  # เปลี่ยน Fully Connected Layer เป็น 5 Classes
model = model.to(device)

# 📌 6. Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📌 7. Train Model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    val_loss, val_acc = 0.0, 0.0

    # 📌 8. Validate Model
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# 📌 9. Save Model
torch.save(model.state_dict(), save_model_path)
print(f"✅ Model saved at {save_model_path}")
