import os
import sys
import torch
import mysql.connector
from PIL import Image
import torchvision.transforms as T
import json
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

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
model.load_state_dict(state_dict, strict=False)

# Define the transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# MySQL connection function
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "embeddings_db"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        sys.exit(1)

def compute_embedding(image_path):
    """Generate embedding for a single image."""
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.extract_features(x).cpu().numpy().flatten()
    return embedding.tolist()

def precompute_embeddings(image_folder):
    """
    Precompute embeddings for all images in a folder and insert them into MySQL.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    inserted_count = 0

    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.abspath(os.path.join(root, file)).replace("/", "\\")

                # Check if the image is already in the database
                cursor.execute("SELECT COUNT(*) FROM embeddings WHERE path = %s", (image_path,))
                if cursor.fetchone()[0] > 0:
                    print(f"Skipping {file}: Already in database.")
                    continue

                try:
                    # Compute embedding for the new image
                    embedding = compute_embedding(image_path)

                    # Insert into MySQL
                    cursor.execute("""
                        INSERT INTO embeddings (filename, path, embedding, url)
                        VALUES (%s, %s, %s, %s)
                    """, (file, image_path, json.dumps(embedding), None))  # Set URL as None for now
                    inserted_count += 1
                    print(f"Inserted {file} into database.")
                except Exception as e:
                    print(f"Error inserting {file}: {e}")

    conn.commit()
    print(f"All changes committed to the database. Total new rows inserted: {inserted_count}")
    cursor.close()
    conn.close()


# Run the script
if __name__ == "__main__":
    # Verify database connection
    conn = get_db_connection()
    if conn.is_connected():
        print("Connected to MySQL!")
    conn.close()

    # Verify database name
    conn = get_db_connection()
    print(f"Connected to database: {os.getenv('DB_NAME')}")

    # Run the embedding precomputation
    image_folder = "wallpaper_data/train"  # Path to your image folder
    precompute_embeddings(image_folder)
