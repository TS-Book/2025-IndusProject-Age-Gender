import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import cv2
import numpy as np

# โหลด Haar Cascade
HAAR_CASCADE_PATH = r"D:\University\3\3_2\Indus based\AGE_Detection\Model\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Define the model class for classification
class AgeModel(nn.Module):
    def __init__(self):
        super(AgeModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 5)  # เปลี่ยนจาก 1 เป็น 5 classes
    
    def forward(self, x):
        return self.model(x)  # ไม่ต้องใช้ softmax ที่นี่ เพราะ CrossEntropyLoss มีอยู่แล้วตอน Train

# Load the model
MODEL_PATH = r"D:\University\3\3_2\Indus based\AGE_Detection\Model\age_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AgeModel().to(DEVICE)

# Load the state_dict and handle key mismatches
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model_state_dict = model.state_dict()

# Filter out unnecessary keys
filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}

# Update the model with the filtered state_dict
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()  # ตั้งเป็นโหมด evaluation

print("Model loaded successfully!")

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ปรับขนาดให้ตรงกับ ResNet18
    transforms.ToTensor(),  # แปลงเป็น Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize ตาม ImageNet
])

# กลุ่มอายุที่โมเดลจำแนก
age_classes = ["Child (0-12)", "Teenager (13-20)", "Adult (20-44)", "Middle age (45-64)", "Aged (65+)"]

# เปิดกล้อง
cap = cv2.VideoCapture(1)  # เปลี่ยนเป็นหมายเลขกล้องที่ต้องการใช้

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงเป็น grayscale เพื่อให้ Haar Cascade ตรวจจับได้แม่นยำขึ้น
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า (scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))

    for (x, y, w, h) in faces:
        # ครอปเฉพาะใบหน้า
        face = frame[y:y+h, x:x+w]

        # แปลงภาพจาก OpenCV (BGR) เป็น PIL (RGB)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(face_rgb)

        # เตรียมภาพสำหรับโมเดล
        input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)  # เพิ่มมิติให้เป็น (1, C, H, W)

        # ทำนายอายุ
        with torch.no_grad():
            output = model(input_tensor)  # ค่าที่ออกมาจะเป็น logits (ไม่ใช่ probability)
            predicted_class = torch.argmax(output, dim=1).item()  # เลือก class ที่มีค่ามากสุด

        predicted_age_group = age_classes[predicted_class]  # Map index ไปที่ชื่อกลุ่มอายุ

        # วาดกรอบสี่เหลี่ยมรอบใบหน้า
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # แสดงผลลัพธ์บนหน้าจอ
        text = f"Age Group: {predicted_age_group}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # แสดงภาพ
    cv2.imshow("Age Detection", frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
