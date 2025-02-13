import os
import sys
import json
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from inference.inference import app  # if inference.py is inside an inference/ folder
except ModuleNotFoundError:
    from inference import app  # if inference.py is in the root directory

client = TestClient(app)

# ✅ 1. Test if the homepage is reachable
def test_homepage():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]  # Ensure HTML response

# ✅ 2. Test searching by SKU name (Text input)
def test_text_search():
    test_sku = "SE II 24K7151"  # UPDATE with SKU design
    response = client.post("/recommendations/filename", json={"filename": test_sku})
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0  # at least one result
    # Validate response structure
    first_result = data["recommendations"][0]
    assert "filename" in first_result
    assert "Product_Type" in first_result
    assert "UOM" in first_result
    assert "inventory_count" in first_result
    assert "similarity" in first_result

# ✅ 3. Test searching with an invalid SKU name
def test_invalid_text_search():
    response = client.post("/recommendations/filename", json={"filename": "INVALID_SKU"})
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "No images found for the provided SKU."

# ✅ 4. Test image upload for similarity search
def test_upload_image_search():
    image_path = os.path.join(os.path.dirname(__file__), "data", "A 2301.jpg")
    assert os.path.exists(image_path), "Sample image for testing is missing!"
    with open(image_path, "rb") as img:
        response = client.post("/recommendations/", files={"file": img})
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0  # ensure results are returned
    # Validate response structure
    first_result = data["recommendations"][0]
    assert "filename" in first_result
    assert "Product_Type" in first_result
    assert "UOM" in first_result
    assert "inventory_count" in first_result
    assert "similarity" in first_result

# ✅ 5. Test uploading an invalid file format (Should fail)
def test_invalid_image_upload():
    invalid_file_path = os.path.join(os.path.dirname(__file__), "sample.txt")
    assert os.path.exists(invalid_file_path), "Sample text file for testing is missing!"
    with open(invalid_file_path, "rb") as f:
        response = client.post("/recommendations/", files={"file": f})
    assert response.status_code == 400
    data = response.json()
    # Expect the error message to mention an invalid image file.
    assert "Invalid image file" in data["detail"]
