from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import shutil, os, json, re

from ocr.pdf_converter import convert_pdf_to_images
from ocr.image_processor import preprocess_all_images
from ocr.handwriting_ocr import extract_text_from_image
from ocr.text_cleaner import clean_results
from database import init_db, get_db, SessionLocal, Upload, ExtractedQuestion

app = FastAPI(title="AI OCR Backend (TrOCR)")

# Mount the processed folder so the frontend can load images
os.makedirs("processed", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
app.mount("/images", StaticFiles(directory="processed"), name="images")

# Allow the frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


# ── Route 1: Health check ─────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "AI OCR Backend (TrOCR) is running!"}


# ── Helper: Separate Questions ────────────────────────────────────────
def format_answer_lines(lines):
    if not lines:
        return ""
    result = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # If line starts with a list marker, rule, or code comment, keep it on a newline
        result.append(l)
    return "\n".join(result).strip()


def separate_questions(raw_results):
    questions = []
    current_q = None
    current_a = []
    current_conf = []

    for item in raw_results:
        text = item["text"].strip()
        conf = item.get("confidence", 0.95)

        # Match "Q# 01", "Q#02:-", "Question 1", "Q1:", "QB 01", etc.
        q_match = re.match(r'^(([Qq][#\s]*\d+|Question\s*\d+|[QGBqgb]\s*#?\s*\d+)[:.-]?\s*)(.*)$', text, re.IGNORECASE)

        if q_match:
            if current_q:
                questions.append({
                    "question": current_q,
                    "answer": format_answer_lines(current_a),
                    "confidence": round(sum(current_conf)/len(current_conf), 2) if current_conf else 0.95
                })
            current_q = q_match.group(1).strip().rstrip(":-. ")
            rest_of_text = q_match.group(3).strip()
            current_a = [rest_of_text] if rest_of_text else []
            current_conf = [conf]
        elif text.endswith('?') and len(text) < 120:
            if current_q:
                questions.append({
                    "question": current_q,
                    "answer": format_answer_lines(current_a),
                    "confidence": round(sum(current_conf)/len(current_conf), 2) if current_conf else 0.95
                })
            current_q = text
            current_a = []
            current_conf = [conf]
        else:
            if current_q is None:
                current_q = "Q# 01"
                current_a.append(text)
                current_conf.append(conf)
            else:
                current_a.append(text)
                current_conf.append(conf)

    if current_q:
        questions.append({
            "question": current_q,
            "answer": format_answer_lines(current_a),
            "confidence": round(sum(current_conf)/len(current_conf), 2) if current_conf else 0.95
        })

    return questions


# ── Route 2: Upload + process a PDF with real-time streaming progress ─
@app.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # 1. Save PDF
    pdf_path = os.path.join("uploads", file.filename)
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def generate():
        try:
            yield json.dumps({"status": "Converting PDF to images...", "progress": 10}) + "\n"
            
            # 2. PDF → images
            image_paths = convert_pdf_to_images(pdf_path, output_folder="processed")
            total_pages = len(image_paths)
            
            yield json.dumps({"status": f"Preprocessing {total_pages} page(s)...", "progress": 25}) + "\n"
            preprocessed_images = preprocess_all_images(image_paths)
            
            # 3. Detect text regions with PaddleOCR and recognize with TrOCR
            raw_ocr_results = []
            for i, path in enumerate(preprocessed_images):
                progress = 25 + int(60 * ((i + 1) / total_pages))
                yield json.dumps({"status": f"Running TrOCR on page {i+1} of {total_pages}...", "progress": progress}) + "\n"
                page_results = extract_text_from_image(path)
                raw_ocr_results.extend(page_results)
                
            yield json.dumps({"status": "Structuring questions and answers...", "progress": 90}) + "\n"
            cleaned_results = clean_results(raw_ocr_results)
            questions = separate_questions(cleaned_results)

            yield json.dumps({"status": "Saving results...", "progress": 95}) + "\n"

            # 4. Save to database
            with SessionLocal() as db:
                upload_record = Upload(
                    filename    = file.filename,
                    total_pages = total_pages
                )
                db.add(upload_record)
                db.commit()
                db.refresh(upload_record)

                for q in questions:
                    db_q = ExtractedQuestion(
                        upload_id  = upload_record.id,
                        question   = q["question"],
                        answer     = q["answer"],
                        confidence = q["confidence"]
                    )
                    db.add(db_q)
                db.commit()

                image_urls = [f"http://127.0.0.1:8000/images/{os.path.basename(p)}" for p in image_paths]

                final_data = {
                    "id":          upload_record.id,
                    "filename":    upload_record.filename,
                    "total_pages": upload_record.total_pages,
                    "questions":   questions,
                    "raw_text":    cleaned_results,
                    "images":      image_urls
                }

                # Save JSON backup
                result_path = os.path.join("results", file.filename + ".json")
                with open(result_path, "w") as f:
                    json.dump(final_data, f, indent=2)

            yield json.dumps({"status": "Done!", "progress": 100, "result": final_data}) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ── Route 3: Get all past uploads ────────────────────────────────────
@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    uploads = db.query(Upload).order_by(Upload.uploaded_at.desc()).all()
    return [
        {
            "id":          u.id,
            "filename":    u.filename,
            "total_pages": u.total_pages,
            "uploaded_at": u.uploaded_at.isoformat()
        }
        for u in uploads
    ]


# ── Route 4: Get one past result by ID ───────────────────────────────
@app.get("/history/{upload_id}")
def get_upload_result(upload_id: int, db: Session = Depends(get_db)):
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found.")

    questions = db.query(ExtractedQuestion)\
                  .filter(ExtractedQuestion.upload_id == upload_id).all()

    return {
        "id":          upload.id,
        "filename":    upload.filename,
        "total_pages": upload.total_pages,
        "uploaded_at": upload.uploaded_at.isoformat(),
        "questions": [
            {
                "question":   q.question,
                "answer":     q.answer,
                "confidence": q.confidence
            }
            for q in questions
        ]
    }
