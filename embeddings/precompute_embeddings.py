import os
import sys
import torch
import mysql.connector
from PIL import Image
import torchvision.transforms as T
import json
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

# Database connection configuration from environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")  
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            database=DB_NAME,
            port=DB_PORT
        )
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Database Connection Error: {err}")
        sys.exit(1)

# Initialize model and transformations
device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.DRV import HybridSimilarityModel 

# Use the Hugging Face API to download the model from the Hugging Face Hub
def download_model_from_huggingface():
    model_name = "asianrynn/DR50V16"  # The model's name on Hugging Face Hub
    model_file = hf_hub_download(repo_id=model_name, filename="best_DRV.pth")
    return model_file

# Load the pre-trained model
model_file_path = download_model_from_huggingface()
model = HybridSimilarityModel(embedding_size=128).to(device)
model.eval()
state_dict = torch.load(model_file_path, map_location=device)
model.load_state_dict(state_dict)

# Define transformation for image preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to compute image embedding
def compute_embedding(image_path):
    """Generate embedding for a single image."""
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(x).cpu().numpy().flatten()
    return embedding.tolist()

# Precompute embeddings and store them in MySQL
def precompute_embeddings(image_folder):
    """
    Precompute embeddings for all images in a folder and insert or update them in MySQL.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    inserted_count = 0
    updated_count = 0
    skipped_count = 0

    # Fetch all SKUs that already exist in the database
    cursor.execute("SELECT SKU FROM embeddings")
    existing_skus = {row[0] for row in cursor.fetchall()}

    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.abspath(os.path.join(root, file)).replace("/", "\\")
                
                # Extract SKU from filename (assumes filename is SKU.jpg)
                sku = os.path.splitext(file)[0].upper()

                if sku in existing_skus:
                    try:
                        # Compute embedding
                        embedding = compute_embedding(image_path)
                        embedding_json = json.dumps(embedding)

                        # Update the embedding if SKU exists
                        cursor.execute("""
                            UPDATE embeddings
                            SET Embeddings = %s
                            WHERE SKU = %s
                        """, (embedding_json, sku))
                        updated_count += 1
                        print(f"🔄 Updated embedding for SKU {sku}.")
                    except Exception as e:
                        print(f"❌ Error processing {file}: {e}")
                        skipped_count += 1
                else:
                    skipped_count += 1
                    print(f"❌ SKU {sku} not found in database, skipping.")

    conn.commit()
    print(f"✅ Embedding process complete: {updated_count} updated, {skipped_count} skipped.")

    cursor.close()
    conn.close()

def main():
    # Check connection to database
    conn = get_db_connection()
    if conn.is_connected():
        print(f"✅ Connected to database: {DB_NAME}")
    conn.close()

    # Use the IMAGE_FOLDER environment variable if set, otherwise use a default path.
    image_folder = os.getenv("IMAGE_FOLDER", r"C:\Users\User\OneDrive\Desktop\WallpaperAndCarpets_Sdn Bhd\Datasets\Processed Wallpaper Dataset\train")
    precompute_embeddings(image_folder)

if __name__ == "__main__":
    main()
