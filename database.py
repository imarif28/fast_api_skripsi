"""
database.py - Koneksi dan model database menggunakan SQLite + SQLAlchemy
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./skripsi_ttd.db")

# Engine SQLAlchemy
# connect_args={"check_same_thread": False} hanya untuk SQLite. Untuk MySQL, argumen ini dihapus.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class DetectionLog(Base):
    """
    Tabel log untuk setiap hasil analisis tanda tangan.
    """
    __tablename__ = "detection_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    image_filename  = Column(String(255), nullable=False)
    detected_classes = Column(Text, nullable=False)        # JSON string: ["huruf_pertama_besar", ...]
    all_confidences  = Column(Text, nullable=True)         # JSON string: {"kelas1": 0.5, ...}
    detections       = Column(Text, nullable=True)         # JSON string mentah YOLO: [{"class": "...", "confidence": 0.9, "bbox": [1,2,3,4]}]
    narrative        = Column(Text, nullable=False)         # Narasi kepribadian lengkap
    created_at       = Column(DateTime, default=datetime.utcnow)


class PersonalityRule(Base):
    """
    Tabel untuk menyimpan narasi kepribadian per kelas deteksi.
    Siap digunakan untuk fitur Admin Panel (CMS) di masa mendatang.
    """
    __tablename__ = "personality_rules"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    feature_name = Column(String(255), unique=True, index=True, nullable=False)
    narrative_text = Column(Text, nullable=False)


def init_db():
    """Buat semua tabel jika belum ada."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency injection untuk mendapatkan sesi database."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
