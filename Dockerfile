# Use Python 3.12 slim image for pandas-ta compatibility
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Note: Installing torch cpu-only version to save space if GPU not needed
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for logs and data
RUN mkdir -p logs data

# Expose port for dashboard
EXPOSE 8082

# Create entrypoint script
RUN echo '#!/bin/bash\n\
    # Start dashboard in background\n\
    python web/dashboard.py &\n\
    \n\
    # Start continuous trading\n\
    python -m src.main continuous\n\
    ' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
CMD ["/app/entrypoint.sh"]
