# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all necessary project files and directories
COPY inference.py /app/
COPY best_ehfrnet.pth /app/
COPY embeddings.json /app/
COPY model /app/model/
COPY blocks /app/blocks/
COPY templates /app/templates/

# Expose the Flask app's default port
EXPOSE 5000

# Command to run the FastAPI app
CMD ["python", "inference.py"]
