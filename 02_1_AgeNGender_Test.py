import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# 📌 1. Load Haar Cascade
HAAR_CASCADE_PATH = r"D:\University\3\3_2\Indus based\AGE_Detection\Model\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Define the model class for age classification
class AgeModel(nn.Module):
    def __init__(self):
        super(AgeModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 5)  # 5 classes for age

    def forward(self, x):
        return self.model(x)

# Define model class for gender classification (using ResNet18 as well)
class GenderModel(nn.Module):
    def __init__(self):
        super(GenderModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)  # 2 classes for gender (Male/Female)

    def forward(self, x):
        return self.model(x)


#📌 2. Load the age prediction model
AGE_MODEL_PATH = r"D:\University\3\3_2\Indus based\AGE_Detection\Model\age_model.pth"
GENDER_MODEL_PATH = r"D:\University\3\3_2\Indus based\AGE_Detection\Model\gender_prediction_wrn.pth"  # Path to your gender model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

age_model = AgeModel().to(DEVICE)
gender_model = GenderModel().to(DEVICE) # Initialize gender model

# Load the state_dict for age model and handle key mismatches
age_checkpoint = torch.load(AGE_MODEL_PATH, map_location=DEVICE)
age_model_state_dict = age_model.state_dict()
filtered_age_state_dict = {k: v for k, v in age_checkpoint.items() if k in age_model_state_dict}
age_model.load_state_dict(filtered_age_state_dict, strict=False)
age_model.eval()

# Load the state_dict for gender model and handle key mismatches
gender_checkpoint = torch.load(GENDER_MODEL_PATH, map_location=DEVICE)
gender_model_state_dict = gender_model.state_dict()
filtered_gender_state_dict = {k: v for k, v in gender_checkpoint.items() if k in gender_model_state_dict}
gender_model.load_state_dict(filtered_gender_state_dict, strict=False)
gender_model.eval()


print("Models loaded successfully!")

#📌 3. Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Age and Gender classes
age_classes = ["Child (0-12)", "Teenager (13-20)", "Adult (21-44)", "Middle age (45-64)", "Aged (65+)"]
gender_classes = ["Male", "Female"]  # Define gender classes


#📌 4. Open camera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(face_rgb)
        input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

        # Age prediction
        with torch.no_grad():
            age_output = age_model(input_tensor)
            predicted_age_class = torch.argmax(age_output, dim=1).item()

            gender_output = gender_model(input_tensor) # Gender prediction
            predicted_gender_class = torch.argmax(gender_output, dim=1).item()


        predicted_age_group = age_classes[predicted_age_class]
        predicted_gender = gender_classes[predicted_gender_class] # Get predicted gender string

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display age and gender on separate lines
        gender_text = f"Gender: {predicted_gender}"
        age_text = f"Age: {predicted_age_group}"

        # Calculate text positions for two lines
        text_y = y - 10  # Start position for the first line
        cv2.putText(frame, gender_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, age_text, (x, text_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Put age above gender


    cv2.imshow("Age and Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()