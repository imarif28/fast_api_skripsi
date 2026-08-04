# ── Stage Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install system deps yang dibutuhkan OpenCV headless & ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies lebih dulu (memanfaatkan Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi
COPY . .

# Buat folder uploads (persistent volume akan di-mount di sini)
RUN mkdir -p uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
