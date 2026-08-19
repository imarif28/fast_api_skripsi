"""
database.py - Koneksi dan model database menggunakan SQLite + SQLAlchemy
Hanya menyimpan tabel personality_rules. DetectionLog dihapus.
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./skripsi_ttd.db")

# Engine SQLAlchemy
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PersonalityRule(Base):
    """
    Tabel untuk menyimpan narasi kepribadian per kelas deteksi.
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
