"""
main.py - FastAPI application untuk analisis kepribadian via tanda tangan
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import uuid
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import init_db, get_db, DetectionLog, SessionLocal
from inference import SignatureDetector
from personality import build_narrative, seed_personality_rules

load_dotenv()

# ==============================================================================
# Startup & Shutdown (lifespan)
# ==============================================================================

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    version="1.0.0",
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
    """
    # ── Validasi format file ─────────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    ext = file.filename.split('.')[-1].lower() if file.filename else ""
    
    if file.content_type not in allowed_types and ext not in {"jpg", "jpeg", "png"}:
        raise HTTPException(
            status_code=400,
            detail=f"Format file tidak didukung: '{file.content_type}'. Gunakan JPG atau PNG.",
        )

    # ── Baca bytes gambar ────────────────────────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File gambar kosong.")

    # ── Simpan file ke sistem (Uploads Directory) ────────────────────────────
    file_ext = ext if ext else "jpg"
    unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    # ── Jalankan inferensi ───────────────────────────────────────────────────
    try:
        detections, all_confs = await asyncio.to_thread(detector.detect, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan saat inferensi: {str(e)}")

    # ── Bangun narasi kepribadian ────────────────────────────────────────────
    # Ambil nama kelas unik yang terdeteksi (deduplikasi untuk narasi)
    detected_class_names = list({d["class_name"] for d in detections})
    narrative = build_narrative(db, detected_class_names)

    # ── Simpan log ke database ───────────────────────────────────────────────
    log_entry = DetectionLog(
        image_filename  = unique_filename,
        detected_classes = json.dumps(detected_class_names, ensure_ascii=False),
        all_confidences  = json.dumps(all_confs, ensure_ascii=False),
        detections       = json.dumps(detections, ensure_ascii=False),
        narrative        = narrative,
        created_at       = datetime.utcnow(),
    )
    db.add(log_entry)
    db.commit()

    # ── Susun respons JSON ───────────────────────────────────────────────────
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


# ==============================================================================
# Endpoint: Riwayat Analisis
# ==============================================================================

@app.get("/history", tags=["Riwayat"])
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    """
    Ambil riwayat analisis terbaru dari database.

    - **limit**: Jumlah data yang ingin diambil (default: 10)
    """
    logs = (
        db.query(DetectionLog)
        .order_by(DetectionLog.created_at.desc())
        .limit(limit)
        .all()
    )

    history = []
    for log in logs:
        history.append({
            "id"             : log.id,
            "image_filename" : log.image_filename,
            "detected_classes": json.loads(log.detected_classes),
            "all_confidences": json.loads(log.all_confidences) if log.all_confidences else {},
            "detections"     : json.loads(log.detections) if getattr(log, 'detections', None) else [],
            "narrative"      : log.narrative,
            "created_at"     : log.created_at.isoformat() if log.created_at else None,
        })

    return {"status": "success", "data": history}


@app.get("/history/{log_id}/image", tags=["Riwayat"])
def get_history_image(log_id: int, db: Session = Depends(get_db)):
    """
    Ambil gambar dari riwayat analisis berdasarkan ID.
    """
    log = db.query(DetectionLog).filter(DetectionLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Riwayat tidak ditemukan.")
    
    file_path = os.path.join(UPLOAD_DIR, log.image_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Gambar fisik tidak ditemukan di server.")
        
    return FileResponse(file_path)


@app.delete("/history/clear", tags=["Riwayat"])
def clear_history(db: Session = Depends(get_db)):
    """
    Menghapus semua riwayat analisis dari database dan file gambar fisik.
    """
    try:
        db.query(DetectionLog).delete()
        db.commit()
        
        # Bersihkan file fisik di direktori uploads
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                
        return {"status": "success", "message": "Semua riwayat berhasil dibersihkan."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membersihkan riwayat: {e}")
