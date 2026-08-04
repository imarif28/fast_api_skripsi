"""
personality.py - Kamus pemetaan kelas deteksi ke narasi kepribadian
"""
import json
import os
from sqlalchemy.orm import Session
from database import PersonalityRule



# ==============================================================================
# Kelas yang boleh memiliki MULTIPLE bounding box
# ==============================================================================
MULTIPLE_BBOX_CLASSES: set[str] = {
    "coretan_badan",
    "pertemuan_garis",
    "ornamen",
    "jarak_kosong",
}

def seed_personality_rules(db: Session):
    """
    Sinkronisasi tabel personality_rules dengan file personality_data.json.
    - Menambah entri yang belum ada di database.
    - Menghapus entri yang sudah tidak ada di JSON (label dihapus dari model).
    Dipanggil saat startup server.
    """
    json_path = os.path.join(os.path.dirname(__file__), "personality_data.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            personality_map: dict = json.load(f)
    except Exception as e:
        print(f"[ERROR] Gagal membaca {json_path}: {e}")
        return

    json_keys = set(personality_map.keys())

    # Hapus entri yang sudah tidak ada di JSON (label dihapus dari model)
    all_rules = db.query(PersonalityRule).all()
    stale = [r for r in all_rules if r.feature_name not in json_keys]
    if stale:
        for r in stale:
            print(f"[INFO] Menghapus narasi usang dari database: '{r.feature_name}'")
            db.delete(r)
        db.commit()

    # Tambah / update entri yang ada di JSON
    existing_keys = {r.feature_name for r in db.query(PersonalityRule).all()}
    new_rules = []
    for feature, text in personality_map.items():
        if feature not in existing_keys:
            new_rules.append(PersonalityRule(feature_name=feature, narrative_text=text))
    
    if new_rules:
        db.add_all(new_rules)
        db.commit()
        print(f"[INFO] Berhasil menambah {len(new_rules)} narasi kepribadian baru ke database.")
    else:
        print(f"[INFO] Database personality_rules sudah sinkron ({len(personality_map)} aturan aktif).")

def build_narrative(db: Session, detected_classes: list[str]) -> str:
    """
    Menyusun narasi kepribadian utuh berdasarkan daftar kelas yang terdeteksi.
    Mengambil data narasi dinamis dari database (Tabel personality_rules).
    
    Format:
    - Narasi disusun secara berurutan sesuai deteksi.
    - Tiap narasi digabung dalam satu paragraf panjang atau dibatasi spasi ganda.
    """
    if not detected_classes:
        return (
            "Tidak ditemukan pola grafologi dominan pada tanda tangan ini. "
            "Namun hal ini wajar dan tidak mengindikasikan masalah psikologis tertentu."
        )

    # Query ke database untuk mengambil narasi berdasarkan daftar kelas
    rules = db.query(PersonalityRule).filter(PersonalityRule.feature_name.in_(detected_classes)).all()
    
    # Buat dictionary hasil query untuk menjaga urutan output jika diperlukan
    db_map = {rule.feature_name: rule.narrative_text for rule in rules}
    
    narratives = []
    # Loop mengikuti urutan di detected_classes
    for cls in detected_classes:
        text = db_map.get(cls)
        if text:
            narratives.append(text)

    # Menggabungkan narasi dengan satu spasi antar kalimat
    if narratives:
        return " ".join(narratives)
    else:
        return "Pola yang terdeteksi belum memiliki interpretasi kepribadian di dalam database."
