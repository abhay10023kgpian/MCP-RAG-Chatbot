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

# Copy project files
COPY . .

# Default port (overridden by docker-compose or Render)
ENV PORT=8000

# Expose API port
EXPOSE ${PORT}

# Default: run the production startup script (embed + uvicorn)
CMD ["python", "start.py"]
