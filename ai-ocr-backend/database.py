from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./ocr_results.db"

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


# ── Table: every uploaded file ────────────────────────────────────────
class Upload(Base):
    __tablename__ = "uploads"

    id          = Column(Integer, primary_key=True, index=True)
    filename    = Column(String,  index=True)
    total_pages = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


# ── Table: every extracted question ──────────────────────────────────
class ExtractedQuestion(Base):
    __tablename__ = "extracted_questions"

    id          = Column(Integer, primary_key=True, index=True)
    upload_id   = Column(Integer, index=True)   # links back to Upload
    question    = Column(String)
    answer      = Column(Text)
    confidence  = Column(Float)


# ── Create tables if they don't exist yet ────────────────────────────
def init_db():
    Base.metadata.create_all(bind=engine)


# ── Helper: get a database session ───────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
