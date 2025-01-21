import os
import sys
import json
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
from urllib.parse import quote, unquote
from torchvision import transforms as T
import uvicorn

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
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Directory of inference.py
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # Root directory of HECT-Net WCD
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "embeddings.json")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "static")
MODEL_PATH = os.path.join(ROOT_DIR, "best_ehfrnet.pth")  # Correct path to model file

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Update the path for model imports
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from model.ehfrnet import EHFRNetMultiScale

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Jinja2 templates setup
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class WallpaperRecommender:
    def __init__(self, model_path=MODEL_PATH, embeddings_path=EMBEDDINGS_PATH, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Initialize model
        self.model = EHFRNetMultiScale(num_classes=10)
        self.model.eval().to(self.device)

        # Load model weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        if "classifier.weight" in state_dict:
            del state_dict["classifier.weight"]
        if "classifier.bias" in state_dict:
            del state_dict["classifier.bias"]
        self.model.load_state_dict(state_dict, strict=False)

        # Transform for input images
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        # Load precomputed embeddings
        self.embeddings = self._load_embeddings(embeddings_path)

    def _load_embeddings(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _compute_embedding(self, img_path):
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.extract_features(x)
        return emb.cpu().numpy().flatten()

    def recommend(self, user_img_path, top_k=6):
        user_emb = self._compute_embedding(user_img_path)
        similarities = []

        for item in self.embeddings:
            stock_emb = np.array(item["embedding"])
            similarity = self._cosine_similarity(user_emb, stock_emb)
            similarities.append({
                "path": item["path"],  # Path to the recommended image
                "filename": item["filename"],  # Name of the recommended image file
                "url": item.get("url", "#"),  # URL for the product page (default to "#")
                "similarity": similarity  # Similarity score
            })

        # Sort by descending similarity and return top_k results
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
    """
    Serve the index.html from the templates folder.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/recommendations/")
async def get_recommendations(file: UploadFile = File(...)):
    """
    Process the uploaded file and return similarity recommendations.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    # Save user-uploaded file into 'uploads'
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Get top recommendations
    recommendations = recommender.recommend(file_path, top_k=6)

    # Format recommendations
    base_url = "http://127.0.0.1:5000"
    formatted_recommendations = []
    for item in recommendations:
        encoded_path = quote(item["path"], safe="")
        formatted_recommendations.append({
            "filename": item["filename"],
            "image_url": f"{base_url}/dataset_image?file={encoded_path}",
            "url": item["url"],  # Include the URL in the response
            "similarity": item["similarity"]
        })

    return {"recommendations": formatted_recommendations}



@app.get("/dataset_image")
async def dataset_image(file: str):
    """
    Serve images directly from the dataset path.
    """
    image_path = unquote(file)
    image_path = os.path.normpath(os.path.abspath(image_path))

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(image_path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
