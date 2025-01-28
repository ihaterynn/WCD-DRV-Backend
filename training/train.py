import os
import sys
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR

# Ensure Python can find the 'models' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ResNetSimilarityModel from resnet.py
from model.resnet import ResNetSimilarityModel

class CustomWallpaperDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_classes=10):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.lower().endswith(('jpg', 'png'))
        ]
        random.seed(42)
        self.labels = [random.randint(0, num_classes - 1) for _ in range(len(self.image_paths))]  # Randomized labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            return None, None


def print_progress_bar(batch_idx, total_batches, loss, accuracy=None):
    """Print a progress bar with the classic verbose style."""
    bar_length = 30
    progress = int((batch_idx / total_batches) * bar_length)
    bar = '=' * progress + '.' * (bar_length - progress)
    acc_text = f" | Accuracy: {accuracy:.2%}" if accuracy is not None else ""
    print(f"\r[{bar}] - Batch {batch_idx}/{total_batches} - Loss: {loss:.4f}{acc_text}", end="")


def main():
    # Hyperparameters
    data_root = r"C:\Users\User\OneDrive\Desktop\Wallpaper&Carpets Sdn Bhd\Datasets\Processed Wallpaper Dataset\train"
    batch_size = 32
    num_epochs = 12
    lr = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms with data augmentation and normalization
    common_tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    train_dataset = CustomWallpaperDataset(data_root, transform=common_tfm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Train Dataset Size: {len(train_dataset)}")

    # Instantiate ResNetSimilarityModel
    model = ResNetSimilarityModel(embedding_size=128).to(device)
    print(model)

    # Loss, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()  # You might want to change this for similarity search
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs by 50%

    # Mixed precision training
    scaler = GradScaler()

    # To save only the best model
    best_train_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # ----------------- TRAIN PHASE -----------------
        model.train()
        running_loss, running_correct, total_samples = 0.0, 0, 0
        total_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                # Forward pass for embeddings
                embeddings = model(images)
                # Here you might need to adjust the loss function for similarity search
                # For example, using a triplet loss or contrastive loss
                loss = criterion(embeddings, labels)  # Adjust this based on your task

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate accuracy (if applicable)
            _, preds = torch.max(embeddings, 1)
            batch_accuracy = (preds == labels).sum().item() / labels.size(0)
            running_correct += batch_accuracy * labels.size(0)
            total_samples += labels.size(0)

            # Accumulate stats
            running_loss += loss.item()

            # Print progress bar
            avg_loss = running_loss / batch_idx
            avg_acc = running_correct / total_samples if total_samples > 0 else 0.0
            print_progress_bar(batch_idx, total_batches, avg_loss, avg_acc)

        train_loss = running_loss / total_batches
        train_acc = running_correct / total_samples
        print(f"\n  Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2%}")

        # Save the best model based on train accuracy
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), "best_resnet.pth")
            print(f"** Saved new best model with Train Accuracy: {best_train_acc:.2%} **")

        # Step the scheduler
        scheduler.step()


if __name__ == "__main__":
    main()