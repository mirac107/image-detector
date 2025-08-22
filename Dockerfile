FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first (new resolver + TLS fixes)
RUN python -m pip install --no-cache-dir --upgrade pip

# Install PyTorch from the fast official CDN (CPU wheels)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1

# Now install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1"]
