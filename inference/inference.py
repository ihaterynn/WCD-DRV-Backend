import os
import sys
import torch
import mysql.connector
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import numpy as np
from urllib.parse import quote, unquote
from torchvision import transforms as T
from dotenv import load_dotenv
import uvicorn
import json
import requests
from io import BytesIO
import traceback
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import hf_hub_download
import torch

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if running inside Docker or local environment
if os.path.exists("/app/best_DRV.pth"):
    # Running inside Docker container
    BASE_DIR = "/app"
    ROOT_DIR = "/app"
    UPLOAD_FOLDER = "/app/uploads"
    TEMPLATES_DIR = "/app/templates"
    STATIC_DIR = "/app/static"
    MODEL_PATH = "/app/best_DRV.pth"
else:
    # Running locally
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
    TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
    STATIC_DIR = os.path.join(ROOT_DIR, "static")
    MODEL_PATH = os.path.join(ROOT_DIR, "best_DRV.pth")

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from model.DRV import HybridSimilarityModel 

templates = Jinja2Templates(directory=TEMPLATES_DIR)

class WallpaperRecommender:
    def __init__(self, model_name="asianrynn/DR50V16", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Fetch Hugging Face Token from environment
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        # Load model from Hugging Face Hub with authentication token
        model_file = hf_hub_download(repo_id=model_name, filename="best_DRV.pth", use_auth_token=huggingface_token)
        self.model = HybridSimilarityModel(embedding_size=128)
        self.model.eval().to(self.device)

        state_dict = torch.load(model_file, map_location=device)
        self.model.load_state_dict(state_dict)

        # Transform for input images
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _compute_embedding(self, img_path):
        try:
            if img_path.startswith("http"):
                try:
                    response = requests.get(img_path)
                    response.raise_for_status()  # Raises an error for bad status codes
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")
            else:
                img = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file provided.")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(x)
        return emb.cpu().numpy().flatten()

    def _get_embeddings_from_db(self, sku=None):
        """
        Fetches columns: SKU, Product_Type, UOM, Inventory_Internal,
        Image_1 ... Image_8, and Embeddings.
        """
        try:
            IS_CI_CD = os.getenv("GITHUB_ACTIONS") == "true"

            if IS_CI_CD:
                DB_HOST = os.getenv("DB_HOST")
                DB_PORT = int(os.getenv("DB_PORT", 3306))
                DB_USER = os.getenv("DB_USER", "test_user")
                DB_PASSWORD = os.getenv("DB_PASSWORD", "test_password")
                DB_NAME = os.getenv("DB_NAME", "test_db")
                print("🔹 Using **CLOUD MySQL (CI/CD Mode)**")
            else:
                DB_HOST = "host.docker.internal" if os.getenv('DOCKER') == 'true' else "srv1630.hstgr.io"
                DB_PORT = int(os.getenv("DB_PORT", 3306))
                DB_USER = os.getenv("DB_USER", "root")
                DB_PASSWORD = os.getenv("DB_PASSWORD", "")
                DB_NAME = os.getenv("DB_NAME")
                print("🖥️ Using **LOCAL MySQL (Development Mode)**")

            conn = mysql.connector.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                port=DB_PORT
            )

            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT SKU, `Product_Type`, UOM, Inventory_Internal, 
                    Image_1, Image_2, Image_3, Image_4, Image_5, Image_6, Image_7, Image_8, Embeddings
                FROM embeddings 
                WHERE SKU NOT LIKE '%_aug'
            """
            if sku:
                query += " AND SKU = %s"
                cursor.execute(query, (sku,))
            else:
                cursor.execute(query)

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            parsed_results = []
            for row in results:
                try:
                    file_path = row["Image_1"]
                    if os.getenv('DOCKER') == 'true':
                        local_path_prefix = "C:/Users/User/OneDrive/Desktop/Wallpaper&Carpets Sdn Bhd/Datasets/Processed Wallpaper Dataset"
                        docker_mount_prefix = "/app/dataset"
                        if file_path.startswith(local_path_prefix):
                            file_path = docker_mount_prefix + file_path[len(local_path_prefix):]

                    file_path = file_path.replace("\\", "/").strip()
                    embedding_data = json.loads(row["Embeddings"])
                    if not embedding_data or (isinstance(embedding_data, dict) and not embedding_data):
                        continue
                    if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                        embedding_data = embedding_data["embedding"]
                    if not isinstance(embedding_data, list) or len(embedding_data) == 0:
                        continue

                    parsed_results.append({
                        "SKU": row["SKU"].strip(),
                        "Product_Type": row["Product_Type"].strip(),
                        "UOM": row["UOM"].strip(),
                        "Inventory_Internal": row["Inventory_Internal"],
                        "Image_1": file_path,
                        "Image_2": row["Image_2"],
                        "Image_3": row["Image_3"],
                        "Image_4": row["Image_4"],
                        "Image_5": row["Image_5"],
                        "Image_6": row["Image_6"],
                        "Image_7": row["Image_7"],
                        "Image_8": row["Image_8"],
                        "Embeddings": embedding_data
                    })
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for SKU {row['SKU']}: {e}")
            return parsed_results

        except mysql.connector.Error as err:
            print(f"Database error: {err}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return []

    def _cosine_similarity(self, a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot / (norm_a * norm_b + 1e-8)

    def recommend(self, user_img_path, top_k=50):
        user_emb = self._compute_embedding(user_img_path)
        embeddings = self._get_embeddings_from_db()
        similarities = []
        for item in embeddings:
            if "_aug" in item['SKU']:
                continue
            stock_emb = np.array(item["Embeddings"])
            similarity = self._cosine_similarity(user_emb, stock_emb)
            similarity_percentage = similarity * 100
            similarities.append({
                "filename": item["SKU"],
                "Product_Type": item["Product_Type"],
                "UOM": item["UOM"],
                "inventory_count": item["Inventory_Internal"],
                "image_url": item["Image_1"] if item["Image_1"].startswith("http") else None,
                "similarity": similarity_percentage
            })
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

# Initialize recommender
recommender = WallpaperRecommender()


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/recommendations/filename")
async def get_recommendations_by_sku(request: Request):
    """
    Accepts a JSON body with key "filename" (the SKU)
    and returns recommendations based on the primary image.
    """
    body = await request.json()
    sku = body.get("filename")
    if not sku:
        raise HTTPException(status_code=400, detail="SKU not provided.")
    all_embeddings = recommender._get_embeddings_from_db(sku)
    if not all_embeddings:
        raise HTTPException(status_code=404, detail="No images found for the provided SKU.")
    input_image_path = all_embeddings[0]["Image_1"]
    input_embedding = recommender._compute_embedding(input_image_path)
    similarities = []
    all_embeddings = recommender._get_embeddings_from_db()  
    for item in all_embeddings:
        if "_aug" in item['SKU']:
            continue
        stock_emb = np.array(item["Embeddings"])
        similarity_percentage = recommender._cosine_similarity(input_embedding, stock_emb) * 100
        similarities.append({
            "filename": item["SKU"],
            "Product_Type": item["Product_Type"],
            "UOM": item["UOM"],
            "inventory_count": item["Inventory_Internal"],
            "image_url": item["Image_1"] if item["Image_1"].startswith("http") else None,
            "similarity": similarity_percentage
        })
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    main_image_url = input_image_path if input_image_path.startswith("http") else f"http://127.0.0.1:8080/dataset_image?file={quote(input_image_path, safe='')}"
    return {
        "recommendations": similarities[:50],
        "image_url": main_image_url
    }


@app.post("/recommendations/")
async def get_recommendations(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    recommendations = recommender.recommend(file_path, top_k=50)
    base_url = "http://127.0.0.1:8080"
    formatted_recommendations = []
    for item in recommendations:
        image_url = item["image_url"] if item["image_url"] is not None else f"{base_url}/dataset_image?file={quote(item['filename'], safe='')}"
        formatted_recommendations.append({
            "filename": item["filename"],
            "Product_Type": item["Product_Type"],
            "UOM": item["UOM"],
            "inventory_count": item["inventory_count"],
            "url": "#",
            "image_url": image_url,
            "similarity": item["similarity"]
        })
    return {"recommendations": formatted_recommendations}


@app.get("/dataset_image")
async def dataset_image(file: str):
    image_path = unquote(file)
    if image_path.startswith("http"):
        return RedirectResponse(url=image_path)
    if os.getenv('DOCKER') == 'true':
        docker_mount_prefix = "/app/dataset"
        local_path_prefix = "C:/Users/User/OneDrive/Desktop/Wallpaper&Carpets Sdn Bhd/Datasets/Processed Wallpaper Dataset"
        if image_path.startswith(local_path_prefix):
            image_path = docker_mount_prefix + image_path[len(local_path_prefix):]
    image_path = os.path.normpath(os.path.abspath(image_path))
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(image_path)

##############################################
#              SCHEDULER SECTION             #
##############################################
def scheduled_task():
    """Run update_db.py then precompute_embeddings.py once every 5 seconds (for testing)."""
    try:
        # Import update_db and precompute_embeddings functions
        from embeddings.update_db import main as update_db  
        from embeddings.precompute_embeddings import main as precompute_embeddings  

        print("🔄 Scheduled task started: Updating database from Google Sheets...")
        update_db()  # Calls update_db.main() to update MySQL
        print("✅ Database update completed.")

        # Wait to ensure the database commits are fully settled
        import time
        print("⏳ Waiting for database commits to settle...")
        time.sleep(5)

        print("🔄 Starting embeddings precomputation...")
        precompute_embeddings()  # Calls precompute_embeddings.main() to compute new embeddings
        print("✅ Precompute embeddings completed.")

    except Exception as e:
        print("❌ Error in scheduled task:", e)
        traceback.print_exc()

# Initialize the APScheduler background scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_task, 'interval', hours=24)  # run every 24 hrs
scheduler.start()
print("⏰ Scheduler started: Task will run every 5 seconds.")


##############################################
#            END SCHEDULER SECTION           #
##############################################

if __name__ == "__main__":
    if os.getenv("RENDER") == "true":
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    else:
        uvicorn.run(app, host="127.0.0.1", port=8080)
