import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from lightly.loss import NTXentLoss  

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------- LOGGING --------------
def get_log_file(log_name="log", ext=".txt"):
    """Generate a unique log file name like log.txt, log2.txt, etc."""
    fname = log_name + ext
    idx = 1
    while os.path.exists(fname):
        idx += 1
        fname = f"{log_name}{idx}{ext}"
    return fname

log_file = get_log_file("log_sim")

def log_message(msg, to_console=True):
    """Log message to file and optionally to console."""
    if to_console:
        print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

def format_time(s):
    """Format seconds into h m s."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    s = int(s % 60)
    return f"{h}h {m}m {s}s"

# -------------- DATASET --------------
class CustomWallpaperDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_classes=10):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(data_dir, p)
            for p in os.listdir(data_dir)
            if p.lower().endswith(('jpg', 'png'))
        ]
        random.seed(42)
        # random labels for demonstration - do not represent real classes
        self.labels = [random.randint(0, num_classes - 1) for _ in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = self.labels[idx]
            return img, label
        except Exception as e:
            log_message(f"Failed to load {path}: {e}")
            return None, None

# -------------- SIMILARITY METRICS --------------

def compute_avg_positive_cosine(embeddings, labels):
    device = embeddings.device
    bs = embeddings.size(0)
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(norm_emb, norm_emb.t())

    cos_sum = 0.0
    count = 0

    for i in range(bs):
        label_i = labels[i].item()
        pos_indices = (labels == label_i).nonzero(as_tuple=True)[0]
        pos_indices = pos_indices[pos_indices != i]
        if len(pos_indices) > 0:
            cos_sum += sim_matrix[i, pos_indices].sum().item()
            count += len(pos_indices)

    return (cos_sum / count) if count > 0 else 0.0

def compute_in_batch_recall_k(embeddings, labels, k=5):
    device = embeddings.device
    bs = embeddings.size(0)
    k = min(k, bs - 1)

    norm_emb = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(norm_emb, norm_emb.t())
    diag_mask = torch.eye(bs, device=device).bool()
    sim_matrix[diag_mask] = -9999  # exclude self-sim

    recall_hit = 0
    valid_count = 0

    for i in range(bs):
        same_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
        same_indices = same_indices[same_indices != i]
        if len(same_indices) == 0:
            continue
        valid_count += 1

        # top-k similarity
        _, topk_idx = sim_matrix[i].topk(k, largest=True)
        if any(idx in topk_idx for idx in same_indices):
            recall_hit += 1

    return recall_hit / valid_count if valid_count > 0 else 0.0

# -------------- TRAINING SCRIPT --------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.DRV import HybridSimilarityModel, SelfSupervisedLoss

def main():
    data_root = r"C:\Users\User\OneDrive\Desktop\WallpaperAndCarpets_Sdn Bhd\Datasets\Processed Wallpaper Dataset\train"
    batch_size = 32
    num_epochs = 8
    lr = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = CustomWallpaperDataset(data_root, transform=tfm)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    log_message(f"Train Dataset Size: {len(dataset)}")

    # from model.DRV import HybridSimilarityModel  # or whichever name you used
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model = HybridSimilarityModel(embedding_size=128).to(device)
    log_message(f"Model: HybridSimilarityModel (DINO+ResNet50)")

    # Replace MSELoss with SelfSupervisedLoss (NT-Xent Loss)
    loss_fn = SelfSupervisedLoss()  # Using NT-Xent Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Use ReduceLROnPlateau for plateau detection
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    scaler = torch.amp.GradScaler()

    best_recall5 = 0.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        total_batches = len(train_loader)
        valid_steps = 0
        sum_cossim = 0.0
        sum_recall5 = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            if images is None:
                continue
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Debugging: Check if images and labels are properly loaded
            log_message(f"Processing batch {batch_idx}/{total_batches}, images shape: {images.shape}")

            with autocast(device.type):
                embeddings = model(images)
                # For NT-Xent, we use two augmentations, here you can create augmented images separately
                augmented_embeddings = embeddings  # For simplicity, this is just the same embeddings
                loss = loss_fn(embeddings, augmented_embeddings)  # Using NT-Xent Loss here

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Debugging: Check loss
            log_message(f"Loss after batch {batch_idx}: {loss.item()}")

            if labels.size(0) > 1:
                cossim = compute_avg_positive_cosine(embeddings, labels)
                r5 = compute_in_batch_recall_k(embeddings, labels, k=5)
                sum_cossim += cossim
                sum_recall5 += r5
                valid_steps += 1

        epoch_loss = running_loss / total_batches
        if valid_steps > 0:
            epoch_cossim = sum_cossim / valid_steps
            epoch_recall = sum_recall5 / valid_steps
        else:
            epoch_cossim = 0.0
            epoch_recall = 0.0

        epoch_time = time.time() - epoch_start
        log_message(f"Epoch {epoch}/{num_epochs} Results:")
        log_message(f"  Loss: {epoch_loss:.4f}")
        log_message(f"  Mean Cosine Similarity: {epoch_cossim:.4f}")
        log_message(f"  Recall@5: {epoch_recall:.2%}")
        log_message(f"  Time: {format_time(epoch_time)}\n")

        # Step the scheduler with the metric we want to track - let's use epoch_recall
        scheduler.step(epoch_recall)

        if epoch_recall > best_recall5:
            best_recall5 = epoch_recall
            torch.save(model.state_dict(), "best_DRV.pth")
            log_message(f"** Saved new best model with Recall@5: {best_recall5:.2%} **")

    total_dur = time.time() - start_time
    log_message("Training Complete!")
    log_message(f"Total Training Time: {format_time(total_dur)}")
    log_message(f"Best Recall@5: {best_recall5:.2%}")

if __name__ == "__main__":
    log_message("Similarity Training Session Started\n------------------------")
    main()
