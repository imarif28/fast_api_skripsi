"""
main.py - FastAPI application untuk analisis kepribadian via tanda tangan
Tidak menyimpan riwayat analisis maupun file gambar ke disk/database.
"""

import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import init_db, get_db, SessionLocal
from inference import SignatureDetector
from personality import build_narrative, seed_personality_rules

load_dotenv()

# ==============================================================================
# Startup & Shutdown (lifespan)
# ==============================================================================

detector: SignatureDetector | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inisialisasi saat server mulai, bersihkan saat server berhenti."""
    global detector
    print("[INFO] Menginisialisasi database...")
    init_db()

    print("[INFO] Memeriksa dan mengisi data kepribadian awal (Seeder)...")
    with SessionLocal() as db:
        seed_personality_rules(db)

    print("[INFO] Memuat model ONNX...")
    detector = SignatureDetector()
    print("[INFO] Server siap menerima request!")
    yield
    print("[INFO] Server dimatikan.")


# ==============================================================================
# FastAPI App
# ==============================================================================

app = FastAPI(
    title="Analisis Kepribadian via Tanda Tangan",
    description=(
        "API untuk mendeteksi pola grafologis pada tanda tangan menggunakan "
        "model YOLOv8s (ONNX) dan memberikan interpretasi kepribadian."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS: Izinkan semua origin (untuk Flutter/Android)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Endpoint: Health Check
# ==============================================================================

@app.get("/health", tags=["System"])
def health_check():
    """Cek apakah server sedang berjalan."""
    from datetime import datetime
    return {
        "status": "ok",
        "message": "Server berjalan dengan baik.",
        "model_loaded": detector is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==============================================================================
# Endpoint: Analisis Tanda Tangan
# ==============================================================================

@app.post("/analyze-signature", tags=["Analisis"])
async def analyze_signature(
    file: UploadFile = File(..., description="File gambar tanda tangan (JPG/PNG)"),
    db: Session = Depends(get_db),
):
    """
    Upload gambar tanda tangan untuk dianalisis pola grafologis-nya.

    - **file**: File gambar (JPG atau PNG)
    - **Return**: Bounding box deteksi + narasi kepribadian
    - **Catatan**: Gambar tidak disimpan ke server setelah diproses.
    """
    # ── Validasi format file ─────────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    ext = file.filename.split('.')[-1].lower() if file.filename else ""

    if file.content_type not in allowed_types and ext not in {"jpg", "jpeg", "png"}:
        raise HTTPException(
            status_code=400,
            detail=f"Format file tidak didukung: '{file.content_type}'. Gunakan JPG atau PNG.",
        )

    # ── Baca bytes gambar (tidak disimpan ke disk) ───────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File gambar kosong.")

    # ── Jalankan inferensi ───────────────────────────────────────────────────
    try:
        detections, all_confs = await asyncio.to_thread(detector.detect, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan saat inferensi: {str(e)}")

    # ── Bangun narasi kepribadian ────────────────────────────────────────────
    detected_class_names = list({d["class_name"] for d in detections})
    narrative = build_narrative(db, detected_class_names)

    # ── Susun respons JSON (tanpa menyimpan apapun ke database/disk) ─────────
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "data": {
                "detections": detections,
                "all_confidences": all_confs,
                "personality_analysis": {
                    "narrative": narrative,
                },
            },
        },
    )
