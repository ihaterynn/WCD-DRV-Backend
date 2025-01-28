import os
import random
import glob
from PIL import Image
import torchvision.transforms as T

def preprocess_full_dataset(data_root, output_root, seed=42):
    """
    1. Reads images from data_root.
    2. Processes and saves them into a single directory for training.
    """
    random.seed(seed)

    # Create main dataset folder
    dataset_root = os.path.join(output_root, "Processed Wallpaper Dataset")
    train_root = os.path.join(dataset_root, "train")
    os.makedirs(train_root, exist_ok=True)

    # Transform for resizing while maintaining aspect ratio
    preprocess_transform = T.Compose([
        T.Resize((224, 448)),  # Resize while preserving aspect ratio
        T.ToTensor()
    ])

    # Gather all images
    image_paths = glob.glob(os.path.join(data_root, "*.*"))
    original_count = len(image_paths)
    print(f"Found {original_count} images in {data_root}")

    def process_and_save(img_path, base_dir):
        """Process and save an image in the specified directory."""
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            img = Image.open(img_path).convert("RGB")
            img = preprocess_transform(img)  # Apply resize/pad transform
            save_path = os.path.join(base_dir, f"{img_name}.jpg")
            img = T.ToPILImage()(img)  # Convert tensor back to PIL Image
            img.save(save_path)
        except Exception as e:
            print(f"Failed to process {img_path}. Error: {e}")

    # Save all images into the train folder
    print("Saving all images into train folder ...")
    for path in image_paths:
        process_and_save(path, train_root)

    # Check final counts
    num_train_files = sum([len(files) for _, _, files in os.walk(train_root)])
    print(f"Finished. Train set: {num_train_files} images.")

if __name__ == "__main__":
    data_root = r"C:\Users\User\OneDrive\Desktop\Wallpaper&Carpets Sdn Bhd\Datasets\Wallpaper Designs"
    output_root = r"C:\Users\User\OneDrive\Desktop\Wallpaper&Carpets Sdn Bhd\Datasets"
    preprocess_full_dataset(data_root=data_root, output_root=output_root, seed=42)
