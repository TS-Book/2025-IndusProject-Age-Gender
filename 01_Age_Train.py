# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:48:41 2025

@author: thana
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ==========================
# 1️⃣ ตั้งค่าพื้นฐาน
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โหลด Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Transform สำหรับรูปภาพ
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ==========================
# 2️⃣ Custom Dataset (ใช้ Haar Cascade)
# ==========================
class AgeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))  # อ่านโฟลเดอร์ 0_Child, 1_Teenager, ...
        self.image_paths = []
        self.labels = []

        # วนลูปโหลดรูปจากโฟลเดอร์ตามคลาส
        for class_folder in self.classes:
            class_label = int(class_folder.split("_")[0])  # ดึง label จากชื่อโฟลเดอร์
            class_path = os.path.join(root_dir, class_folder)

            for img_file in os.listdir(class_path):
                if img_file.endswith(".jpg") or img_file.endswith(".png"):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(class_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # แปลงเป็น RGB
        label = self.labels[idx]

        # ตรวจจับใบหน้าด้วย Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]  # ใช้ใบหน้าแรกที่เจอ
            face_crop = image[y:y+h, x:x+w]
            face_pil = Image.fromarray(face_crop)
        else:
            face_pil = Image.fromarray(image)  # ถ้าไม่เจอใบหน้า ใช้รูปเต็มแทน

        image = transform(face_pil)  # Resize & Normalize
        return image, label

# ==========================
# 3️⃣ โหลด DataLoader
# ==========================
train_dataset = AgeDataset("dataset")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================
# 4️⃣ โมเดล WRN-16-8 + DMTL (Pretrained)
# ==========================
class AgeClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(AgeClassifier, self).__init__()
        self.backbone = models.wide_resnet50_2(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)  # เปลี่ยน Fully Connected Layer

    def forward(self, x):
        return self.backbone(x)

# ==========================
# 5️⃣ เทรนโมเดล
# ==========================
def train_model():
    model = AgeClassifier(num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
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
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

    # บันทึกโมเดล
    torch.save(model.state_dict(), "age_detection_model.pth")
    print("✅ โมเดลถูกบันทึกแล้ว!")

# ==========================
# 🔥 เริ่มเทรน
# ==========================
if __name__ == "__main__":
    train_model()

