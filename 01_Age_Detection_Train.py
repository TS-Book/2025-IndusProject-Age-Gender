# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:48:41 2025

@author: thana
"""

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

# 📌 ค่าพื้นฐาน
IMG_SIZE = 224  # Resize หลังจาก Crop
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Haar Cascade สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 📌 แปลงอายุเป็น 5 Classes
AGE_CLASSES = ["Child", "Teenager", "Adult", "Middle Age", "Aged"]

def age_to_class(age):
    if age <= 12:
        return 0  # Child
    elif 13 <= age <= 20:
        return 1  # Teenager
    elif 21 <= age <= 44:
        return 2  # Adult
    elif 45 <= age <= 64:
        return 3  # Middle Age
    else:
        return 4  # Aged

# 📌 Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 📌 Custom Dataset Class
class AgeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = [f for f in os.listdir(root_dir) if f.endswith(".jpg") or f.endswith(".png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ดึงอายุจาก filename (ตัวอย่าง: "25_image.jpg" -> อายุ 25)
        age = int(self.images[idx].split("_")[0])
        label = age_to_class(age)

        # ใช้ Haar cascade ตรวจจับใบหน้า
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]  # ใช้ใบหน้าแรกที่เจอ
            face_crop = image[y:y+h, x:x+w]  # Crop ใบหน้า
            face_pil = Image.fromarray(face_crop)  # Convert to PIL
        else:
            face_pil = Image.fromarray(image)  # ถ้าไม่มีใบหน้า ใช้ภาพเต็ม

        image = transform(face_pil)  # Resize & Normalize
        return image, label

# 📌 โหลด Dataset
train_dataset = AgeDataset("path_to_train_images")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 📌 สร้าง Model WRN-16-8 + DMTL
class WRN16_8(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(WRN16_8, self).__init__()
        self.model = models.wide_resnet50_2(pretrained=True)  # ใช้ WRN-50-2 แทน
        self.model.fc = nn.Linear(2048, num_classes)  # เปลี่ยน Fully Connected Layer

    def forward(self, x):
        return self.model(x)

# 📌 สร้างโมเดล
model = WRN16_8().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📌 Train Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

# 📌 Train the model
train()
