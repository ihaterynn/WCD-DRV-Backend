# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary project files and directories
COPY inference.py /inference
COPY best_ehfrnet.pth .
COPY model/ model/
COPY blocks/ blocks/
COPY templates/ templates/

# Expose the FastAPI app's default port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
