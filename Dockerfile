# 🚀 Rocket Landing AI — Dockerfile
# Multi-purpose container for training, evaluation, and hosting Gradio app
FROM python:3.10-slim

# Install system dependencies needed for Box2D and video rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose port for Gradio web application
EXPOSE 7860

# Default command launches the web simulator
CMD ["python", "app.py"]
