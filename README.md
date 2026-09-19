# 📝 AI-Powered Handwriting OCR & Exam Digitization System

An end-to-end, multi-tier AI handwriting recognition and document structuring pipeline designed to transcribe, format, and structure messy handwritten student answer sheets and exam papers.

---

## 🌟 Key Features

* **Adaptive Vision Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization), bilateral filtering, and dynamic multi-scale resolution scaling to enhance faint pencil and pen strokes.
* **Hybrid OCR Architecture**:
  * **PaddleOCR (DBNet + SVTR)** for high-sensitivity multi-scale text line detection.
  * **Microsoft TrOCR (ViT + RoBERTa)** for deep handwriting character recognition with GPU batching.
* **Geometric Vertical Overlap Sorting**: Uses bounding-box vertical overlap calculations ($\ge 45\%$) to maintain natural reading order.
* **Contextual AI Post-Correction**: Integrates local **Ollama (Qwen 2.5)** with a 7-rule post-processing prompt to fix typos, restore punctuation, and preserve technical formulas and code.
* **Offline Fallback**: Automatic fallback to **SymSpell** compound spelling correction if the local LLM is offline.
* **Full-Stack Web Interface**: **Next.js 16 + Tailwind CSS** frontend with real-time upload progress, question-answer extraction cards, confidence scoring, and side-by-side original vs. extracted view.

---

## 🏗️ System Architecture

```
[ Scanned Handwritten PDF / Image ]
                 │
                 ▼
[ 1. Preprocessing (CLAHE, Bilateral Denoising, High-Res Scaling) ]
                 │
                 ▼
[ 2. Hybrid OCR Engine (PaddleOCR DBNet + Microsoft TrOCR) ]
                 │
                 ▼
[ 3. Geometric Line Clustering (Reading Order Reassembly) ]
                 │
                 ▼
[ 4. Semantic AI Post-Correction (Ollama / Qwen 2.5) ]
                 │
                 ▼
[ 5. Structured Q&A Output + Frontend Dashboard ]
```

---

## 🚀 Quick Start Guide

### Prerequisites
* **Python 3.10+** (Tested on Python 3.11)
* **Node.js 18+** & npm
* **Ollama** (optional, recommended for local LLM post-correction): [Download Ollama](https://ollama.ai)

---

### 1. Backend Setup

```bash
cd ai-ocr-backend

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) For CUDA GPU acceleration:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Setup environment variables
cp .env.example .env

# Start FastAPI server
uvicorn app:app --reload --port 8000
```

---

### 2. Ollama Setup (AI Post-Correction)

```bash
# Pull the recommended model
ollama pull qwen2.5:1.5b

# Optional larger models for 8GB+ VRAM:
# ollama pull qwen2.5:3b
# ollama pull qwen2.5:7b
```

---

### 3. Frontend Setup

```bash
cd ai-ocr-frontend

# Install dependencies
npm install

# Start Next.js development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🔌 API Endpoints

### `POST /upload`
Uploads and processes a multi-page handwritten PDF.

* **Payload**: `multipart/form-data` with `file: <your_file.pdf>`
* **Response**:
```json
{
  "total_pages": 1,
  "images": ["http://localhost:8000/images/page_1.png"],
  "questions": [
    {
      "question": "Q# 01",
      "answer": "Large Language Models (LLMs) are AI models trained on large amounts of text...",
      "confidence": 0.98
    }
  ],
  "raw_text": [
    {"text": "Q# 01", "confidence": 0.98},
    {"text": "Large Language Models (LLMs) are AI models...", "confidence": 0.98}
  ]
}
```

---

## 🛠️ Tech Stack

* **Vision & Image Processing**: OpenCV, PyMuPDF (fitz), NumPy
* **Deep Learning OCR**: Microsoft TrOCR, PaddleOCR, PyTorch (CUDA supported)
* **NLP & Post-Correction**: Ollama (Qwen 2.5), SymSpell
* **Backend**: FastAPI, Uvicorn, SQLAlchemy, SQLite
* **Frontend**: Next.js 16, React 19, Tailwind CSS, Lucide Icons

---

## 📄 License
This project is licensed under the MIT License.
