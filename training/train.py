import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Make sure Python can find the 'models' folder, if it's one level up
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import EHFRNetMultiScale from the multi-file architecture
from model.ehfrnet import EHFRNetMultiScale

def main():
    # hyperparameters
    data_root = "./wallpaper_data"  
    batch_size = 8
    num_epochs = 5
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms (assume images are already resized in preprocess)
    common_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Use ImageFolder (requires subfolders; e.g. same_class or multiple classes)
    train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=common_tfm)
    val_dataset   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=common_tfm)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Number of classes is inferred from the subfolders in train_dataset
    num_classes = len(train_dataset.classes)
    print(f"Classes discovered: {train_dataset.classes} (num_classes={num_classes})")

    # Instantiate EHFRNetMultiScale with the discovered number of classes
    model = EHFRNetMultiScale(num_classes=num_classes).to(device)
    print(model)

    # Standard classification loss and Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # ----------------- TRAIN PHASE -----------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate stats
            running_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ----------------- VALIDATION PHASE -----------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, pred = outputs.max(1)
                val_correct += pred.eq(labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_ehfrnet.pth")
            print("** Saved new best model! **")

if __name__ == "__main__":
    main()
