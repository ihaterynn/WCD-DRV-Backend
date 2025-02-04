
import os
import sys
import torch
import mysql.connector
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
from urllib.parse import quote, unquote
from torchvision import transforms as T
from dotenv import load_dotenv
import uvicorn
import json

# Load environment variables from the .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "static")
MODEL_PATH = os.path.join(ROOT_DIR, "best_resnet.pth")  # Updated model path

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Update the path for model imports
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from model.resnet import ResNetSimilarityModel  # Import the new ResNet model

# Serve static files
## app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Jinja2 templates setup
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class WallpaperRecommender:
    def __init__(self, model_path=MODEL_PATH, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Initialize model
        self.model = ResNetSimilarityModel(embedding_size=128)  # Initialize the new model
        self.model.eval().to(self.device)

        # Load model weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)  # Load the new model's weights

        # Transform for input images
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add normalization
        ])

    def _compute_embedding(self, img_path):
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(x)  # Use the new model's forward pass
        return emb.cpu().numpy().flatten()

    def _get_embeddings_from_db(self, filename=None):
        try:
            # Connect to the MySQL database
            conn = mysql.connector.connect(
                host=os.getenv("DB_HOST", "localhost"),
                user=os.getenv("DB_USER", "root"),
                password=os.getenv("DB_PASSWORD", ""),
                database=os.getenv("DB_NAME", "embeddings_db_resnet"),  # Ensure you're using the correct database
                port=int(os.getenv("DB_PORT", 3306))
            )
            print("Connected to the database successfully.\n")

            cursor = conn.cursor(dictionary=True)

            # Check the current database
            cursor.execute("SELECT DATABASE();")
            current_db = cursor.fetchone()
            print(f"Currently connected to database: {current_db}\n")

            # Query to fetch base images, excluding augmented images, and include inventory_count
            query = """
                SELECT filename, path, embedding, url, inventory_count
                FROM embeddings 
                WHERE filename NOT LIKE '%_aug'
            """
            # If filename is provided, filter results by filename
            if filename:
                # Using parameterized query to prevent SQL injection
                query += " AND filename = %s"
                cursor.execute(query, (filename,))
            else:
                cursor.execute(query)

            results = cursor.fetchall()

            # Print the results row by row for debugging
            print("Fetched rows from database:")
            for row in results:
                print(f"Filename: {row['filename']}, URL: {row['url']}, Inventory: {row['inventory_count']}")

            # Close the connection
            cursor.close()
            conn.close()

            # Parse and return embeddings along with inventory_count
            parsed_results = []
            for row in results:
                try:
                    parsed_results.append({
                        "filename": row["filename"],
                        "path": row["path"],
                        "embedding": json.loads(row["embedding"]),
                        "url": row["url"],
                        "inventory_count": row["inventory_count"]  # Include inventory_count in the result
                    })
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for row {row['filename']}: {e}")

            # Debugging: Check the number of results after filtering
            print(f"Number of images returned after filtering: {len(parsed_results)}")

            return parsed_results

        except mysql.connector.Error as err:
            print(f"Database error: {err}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return []

    def recommend(self, user_img_path, top_k=30):
        user_emb = self._compute_embedding(user_img_path)
        embeddings = self._get_embeddings_from_db()
        similarities = []

        print(f"Fetched {len(embeddings)} base images for similarity comparison.")

        for item in embeddings:
            if "_aug" in item['filename']:
                continue

            stock_emb = np.array(item["embedding"])
            similarity = self._cosine_similarity(user_emb, stock_emb)

            # Convert similarity to percentage
            similarity_percentage = similarity * 100

            print(f"Processing: {item['filename']}, Similarity: {similarity_percentage:.2f}%")

            similarities.append({
                "path": item["path"],
                "filename": item["filename"],
                "url": item.get("url", "#"),
                "inventory_count": item["inventory_count"],  # Include inventory count in the response
                "similarity": similarity_percentage  # Store the percentage value
            })

        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]

    def get_recommendations_by_filename(self, filename, top_k=30):
        print(f"Fetching recommendations for filename: {filename}")  # Debugging log
        # Fetch the exact matching filename and then compute similarity
        embeddings = self._get_embeddings_from_db(filename=filename)
        if not embeddings:
            return []

        similarities = []
        for item in embeddings:
            stock_emb = np.array(item["embedding"])
            similarity_percentage = 100  # Exact match, so similarity is 100%

            similarities.append({
                "path": item["path"],
                "filename": item["filename"],
                "url": item.get("url", "#"),
                "inventory_count": item["inventory_count"],
                "similarity": similarity_percentage
            })

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    @staticmethod
    def _cosine_similarity(a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot / (norm_a * norm_b + 1e-8)


# Initialize recommender
recommender = WallpaperRecommender()


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/recommendations/filename")
async def get_recommendations_by_filename(request: Request):
    body = await request.json()
    filename = body.get("filename")

    if not filename:
        raise HTTPException(status_code=400, detail="Filename not provided.")
    
    # Fetch the image path for the given filename from the database
    embeddings = recommender._get_embeddings_from_db(filename=filename)
    
    if not embeddings:
        raise HTTPException(status_code=404, detail="Filename not found or unavailable.")
    
    # Get the image path for the found filename (use the first match)
    image_path = embeddings[0]["path"]
    
    # Now compute the embedding for this image
    filename_embedding = recommender._compute_embedding(image_path)
    similarities = []

    # Now calculate the similarity of this filename's embedding to others in the database
    for item in embeddings:
        stock_emb = np.array(item["embedding"])
        similarity_percentage = recommender._cosine_similarity(filename_embedding, stock_emb) * 100

        # Ensure the image URL is correctly included in the recommendations
        base_url = "http://127.0.0.1:8000"  # Make sure this is the correct base URL for your setup
        encoded_path = quote(item["path"], safe="")  # Encode path for URL
        similarities.append({
            "path": item["path"],
            "filename": item["filename"],
            "image_url": f"{base_url}/dataset_image?file={encoded_path}",  # Image URL from database
            "url": item.get("url", "#"),
            "inventory_count": item["inventory_count"],
            "similarity": similarity_percentage
        })

    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    # Return the recommendations along with the image URL to display in the frontend
    return {"recommendations": similarities, "image_url": f"{base_url}/dataset_image?file={encoded_path}"}

@app.post("/recommendations/")
async def get_recommendations(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    recommendations = recommender.recommend(file_path, top_k=30)

    base_url = "http://127.0.0.1:8000"
    formatted_recommendations = []
    for item in recommendations:
        encoded_path = quote(item["path"], safe="")  # Encode path
        formatted_recommendations.append({
            "filename": item["filename"],
            "image_url": f"{base_url}/dataset_image?file={encoded_path}",
            "url": item["url"],
            "inventory_count": item["inventory_count"],
            "similarity": item["similarity"]
        })

    return {"recommendations": formatted_recommendations}


@app.get("/dataset_image")
async def dataset_image(file: str):
    image_path = unquote(file)
    image_path = os.path.normpath(os.path.abspath(image_path))

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(image_path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)