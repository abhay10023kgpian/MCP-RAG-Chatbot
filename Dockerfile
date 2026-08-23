FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for chromadb's sqlite)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching — only rebuilds if deps change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (includes chroma_db/ baked in from git)
COPY . .

# Render sets PORT env var; default to 10000 for Render free tier
ENV PORT=10000

# Expose API port
EXPOSE ${PORT}

# Start uvicorn directly — chroma_db is baked in, no embedding needed
CMD uvicorn api.server:app --host 0.0.0.0 --port ${PORT}
