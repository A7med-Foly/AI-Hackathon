# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    TRANSFORMERS_CACHE=/tmp/hf_cache

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files, pre-built Chroma vector DB, and processed JSON metadata
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY chroma_db/ ./chroma_db/
COPY README.md .

# Create cache directories with appropriate permissions
RUN mkdir -p /tmp/hf_cache && chmod -R 777 /tmp/hf_cache

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Command to run the FastAPI server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
