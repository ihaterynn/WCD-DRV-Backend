import os
import sys
import json
from PIL import Image
import torch
import torchvision.transforms as T

# Update the path for model imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.ehfrnet import EHFRNetMultiScale

# Initialize your model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = EHFRNetMultiScale(num_classes=10).to(device)
model.eval()

# Load the state_dict and handle incompatible keys
state_dict = torch.load("best_ehfrnet.pth", map_location=device)
if "classifier.weight" in state_dict:
    del state_dict["classifier.weight"]
if "classifier.bias" in state_dict:
    del state_dict["classifier.bias"]
model.load_state_dict(state_dict, strict=False)  # Allow partial loading

# Define the transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def compute_embedding(image_path):
    """Generate embedding for a single image."""
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.extract_features(x).cpu().numpy().flatten()
    return embedding.tolist()

def precompute_embeddings(image_folder, output_file="embeddings.json"):
    """
    Precompute embeddings for all images in a folder and update the embeddings.json file efficiently.
    - Processes only new images.
    - Appends new embeddings to the existing file.
    """
    # Load existing embeddings if the file exists
    existing_embeddings = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_embeddings = {item["path"]: item for item in json.load(f)}
    
    new_embeddings = []  # To store new embeddings
    processed_files = set(existing_embeddings.keys())  # Track already processed images

    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.abspath(os.path.join(root, file))
                if image_path in processed_files:
                    print(f"Skipping {file}: Already processed.")
                    continue
                
                try:
                    # Compute embedding for the new image
                    embedding = compute_embedding(image_path)
                    new_embeddings.append({
                        "filename": file,
                        "path": image_path,
                        "embedding": embedding
                    })
                    print(f"Processed {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    # Merge new embeddings with existing ones
    all_embeddings = list(existing_embeddings.values()) + new_embeddings

    # Save the updated embeddings to the JSON file
    with open(output_file, "w") as f:
        json.dump(all_embeddings, f, indent=4)
    print(f"Updated embeddings saved to {output_file}")

# Run the script
if __name__ == "__main__":
    image_folder = "wallpaper_data/train"  # Path to your image folder
    output_file = "embeddings.json"  # Path to the embeddings file

    precompute_embeddings(image_folder, output_file)
