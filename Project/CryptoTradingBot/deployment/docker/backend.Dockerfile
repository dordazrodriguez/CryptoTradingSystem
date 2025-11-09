# Backend Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application code to reduce context and avoid frontend
COPY backend/ backend/
COPY data/ data/
COPY trading_engine/ trading_engine/
COPY ml_models/ ml_models/
COPY config/ config/
COPY core/ core/
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY main.py main.py
COPY train_model.py train_model.py
COPY deployment/docker/entrypoint.sh deployment/docker/entrypoint.sh
RUN chmod +x deployment/docker/entrypoint.sh

# Create necessary directories
RUN mkdir -p data/logs data/db

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=backend/app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Health check using Python (NO curl needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()" || exit 1

# Run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
