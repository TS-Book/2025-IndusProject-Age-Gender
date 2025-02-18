import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
from multiprocessing import freeze_support

try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    print("tqdm is not installed. Progress bars will not be displayed.")
    print("Install it with: pip install tqdm  or  conda install -c conda-forge tqdm")
    use_tqdm = False

def create_data_loaders(dataset_path, batch_size, num_workers=4):
    """Creates and returns training and validation data loaders."""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Trains the model and prints training/validation statistics."""

    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loop = train_loader
        if use_tqdm:
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]",
                              mininterval=1.0, miniters=10)

        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():  # Use autocast without device_type
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.empty_cache()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_acc = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_loop = val_loader
            if use_tqdm:
                val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]",
                                mininterval=1.0, miniters=10)

            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        scheduler.step(avg_val_loss)


if __name__ == '__main__':
    freeze_support()

    # --- Configuration ---
    dataset_path = r"D:\University\3\3_2\Indus based\AGE_Detection\Datasets\Dataset_All"
    save_model_path = r"D:\University\3\3_2\Indus based\AGE_Detection\Model\age_model.pth"
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    num_workers = 4

    # --- Data Loaders ---
    train_loader, val_loader = create_data_loaders(dataset_path, batch_size, num_workers=num_workers)

    # --- Model Definition ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")

    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)

    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Train and Save ---
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")