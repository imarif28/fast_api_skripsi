import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./skripsi_ttd.db")
engine = create_engine(DATABASE_URL)

try:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS detection_logs"))
    print("Table detection_logs dropped successfully.")
except Exception as e:
    print(f"Error dropping table: {e}")
