# FastAPI Skripsi - Backend Analisis Grafologi

Ini adalah backend API untuk aplikasi analisis grafologi tanda tangan menggunakan FastAPI dan YOLOv8.

## Fitur
- Deteksi pola tanda tangan dengan model ONNX YOLOv8
- Penentuan profil kepribadian berdasarkan grafologi
- API Endpoint untuk aplikasi mobile (Flutter)

## Cara Menjalankan
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan server
uvicorn main:app --host 0.0.0.0 --port 8000
```
