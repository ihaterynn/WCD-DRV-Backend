import os
import random
import glob
from PIL import Image
import torchvision.transforms as T

def preprocess_and_split(
    data_root,
    output_root,
    desired_total=600,
    train_ratio=0.8,
    seed=42
):
    """
    1. Reads images from data_root (e.g. 305 images).
    2. Ensures total images ~ desired_total (600) by augmenting a subset.
    3. Splits into train & val sets (default 80/20).
    4. Saves them to output_root/train/same_class and output_root/val/same_class.
    """

    random.seed(seed)

    # Directories
    train_dir = os.path.join(output_root, "train", "same_class")
    val_dir   = os.path.join(output_root, "val", "same_class")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Basic augmentation transforms
    augment_transform = T.Compose([
        T.RandomRotation(degrees=30),
        T.RandomHorizontalFlip(p=0.5),
    ])
    base_transform = T.Compose([T.Resize((224, 224))])

    # Gather all images
    image_paths = glob.glob(os.path.join(data_root, "*.*"))
    original_count = len(image_paths)
    print(f"Found {original_count} images in {data_root}")

    # Figure out how many new images we need
    needed_aug = max(0, desired_total - original_count)
    print(f"Augmenting {needed_aug} additional images to reach ~{desired_total} total...")

    if needed_aug > 0 and original_count > 0:
        if needed_aug <= original_count:
            augment_candidates = random.sample(image_paths, k=needed_aug)
        else:
            augment_candidates = random.choices(image_paths, k=needed_aug)
    else:
        augment_candidates = []

    # Shuffle to randomize train/val
    random.shuffle(image_paths)
    random.shuffle(augment_candidates)

    train_count = int(len(image_paths) * train_ratio)
    train_base_paths = image_paths[:train_count]
    val_base_paths   = image_paths[train_count:]

    def process_and_save(img_path, out_dir, suffix="_base"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            img = Image.open(img_path).convert("RGB")
            img = base_transform(img)
            save_path = os.path.join(out_dir, f"{img_name}{suffix}.jpg")
            img.save(save_path)
        except Exception as e:
            print(f"Failed to process {img_path}. Error: {e}")

    def augment_and_save(img_path, out_dir, suffix="_aug"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            img = Image.open(img_path).convert("RGB")
            img = augment_transform(img)
            img = base_transform(img)
            save_path = os.path.join(out_dir, f"{img_name}{suffix}.jpg")
            img.save(save_path)
        except Exception as e:
            print(f"Failed to augment {img_path}. Error: {e}")

    # Save base images
    print("Saving BASE images into train/val ...")
    for path in train_base_paths:
        process_and_save(path, train_dir)
    for path in val_base_paths:
        process_and_save(path, val_dir)

    # Distribute augmented images with same ratio
    if len(augment_candidates) > 0:
        aug_train_count = int(len(augment_candidates) * train_ratio)
        train_aug_paths = augment_candidates[:aug_train_count]
        val_aug_paths   = augment_candidates[aug_train_count:]

        print(f"Saving {len(train_aug_paths)} augmented images in train, "
              f"{len(val_aug_paths)} in val ...")

        for path in train_aug_paths:
            augment_and_save(path, train_dir)
        for path in val_aug_paths:
            augment_and_save(path, val_dir)

    # Check final counts
    num_train_files = len(os.listdir(train_dir))
    num_val_files   = len(os.listdir(val_dir))
    total_final = num_train_files + num_val_files
    print(f"Finished. Train set: {num_train_files} images | "
          f"Val set: {num_val_files} images | Total: {total_final}")

if __name__ == "__main__":
    data_root = r"C:\Users\User\OneDrive\Desktop\Wallpaper&Carpets Sdn Bhd\Datasets\Rezised Wallpapers"
    output_root = "./wallpaper_data"
    preprocess_and_split(
        data_root=data_root,
        output_root=output_root,
        desired_total=600,
        train_ratio=0.8,
        seed=42
    )
