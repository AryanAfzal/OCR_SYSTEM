# 📝 AI-Powered Handwriting OCR & Exam Digitization System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.9.1-006699?style=flat)](https://github.com/PaddlePaddle/PaddleOCR)
[![HuggingFace](https://img.shields.io/badge/TrOCR-microsoft%2Ftrocr--base-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/microsoft/trocr-base-handwritten)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen%202.5-000000?style=flat&logo=ollama&logoColor=white)](https://ollama.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16.2.6-black?style=flat&logo=next.js&logoColor=white)](https://nextjs.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-4.0-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end, multi-tier AI handwriting recognition and document structuring system designed to transcribe, clean, restore, and format unconstrained handwritten student answer sheets and exam papers into structured digital Question-and-Answer pairs with confidence metrics.**

---

## 📑 Table of Contents

- [Problem Statement \& Motivation](#-problem-statement--motivation)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Deep-Dive: How the Pipeline Works](#-deep-dive-how-the-pipeline-works)
  - [Stage 1: Adaptive Preprocessing \& Normalization](#stage-1-adaptive-preprocessing--normalization)
  - [Stage 2: Hybrid OCR Engine (PaddleOCR + TrOCR)](#stage-2-hybrid-ocr-engine-paddleocr--trocr)
  - [Stage 3: Geometric Line Clustering \& Reading Order](#stage-3-geometric-line-clustering--reading-order)
  - [Stage 4: Semantic LLM Post-Correction (Ollama)](#stage-4-semantic-llm-post-correction-ollama)
  - [Stage 5: Question/Answer Parsing \& Disambiguation](#stage-5-questionanswer-parsing--disambiguation)
- [Repository Directory Structure](#-repository-directory-structure)
- [Tech Stack \& Exact Library Versions](#-tech-stack--exact-library-versions)
- [Installation \& Quick Start](#-installation--quick-start)
  - [1-Click Setup (Windows)](#1-click-setup-windows---recommended)
  - [1-Click Setup (Linux / macOS)](#1-click-setup-linux--macos)
  - [Manual Step-by-Step Setup](#manual-step-by-step-setup)
- [Hardware Acceleration \& Performance Benchmarks](#-hardware-acceleration--performance-benchmarks)
- [REST API Reference](#-rest-api-reference)
- [Troubleshooting \& Common Pitfalls](#-troubleshooting--common-pitfalls)
- [Future Roadmap \& FYP Extensions](#-future-roadmap--fyp-extensions)
- [License \& Acknowledgements](#-license--acknowledgements)

---

## 🎯 Problem Statement & Motivation

Digitizing handwritten exam sheets and student notebooks has historically been one of the most difficult challenges in Computer Vision and Natural Language Processing. Traditional OCR solutions (such as Tesseract or basic CRNN engines) fail severely on real-world handwriting due to:

1. **Irregular Cursive \& Faint Strokes**: Variable pen pressures, pencil handwriting, and ruled notebook lines cause line detection models to drop faint characters or falsely identify ruled lines as text.
2. **Aspect Ratio Squishing in Vision Transformers**: Feeding high-aspect-ratio handwritten lines (e.g., $2400 \times 80\text{ px}$, an aspect ratio of $30:1$) directly into Vision Transformers like TrOCR squishes entire sentences into illegible pixel strips.
3. **Scrambled Reading Order**: Students often write multi-column answers, margins, or uneven lines. Centroid-distance grouping merges vertically adjacent lines together into unreadable text jumbles.
4. **Lack of Semantic Understanding**: Pure vision models cannot distinguish between visual character ambiguities (e.g., `0ver` vs `over`, `Jorge` vs `Large`, `teild` vs `field`, `1azy` vs `lazy`).
5. **Loss of Document Structure**: Existing OCR systems dump unformatted text blobs with collapsed whitespace, destroying bullet points, rule numbers, and code comments.

### **The Solution**
This project resolves these challenges by introducing a **3-tier hybrid pipeline**:
* **Tier 1 (Vision Enhancement)**: CLAHE contrast equalisation, dynamic resolution scaling, and bilateral edge preservation.
* **Tier 2 (Hybrid Neural OCR)**: PaddleOCR DBNet for sensitive line detection + Microsoft TrOCR (Vision Transformer) for cursive recognition with parallel CUDA tensor batching.
* **Tier 3 (Semantic Contextual Restoration)**: Local LLM post-processing using Ollama (Qwen 2.5) with strict anti-hallucination guardrails, backed by SymSpell offline fallback.

---

## 🌟 Key Features

* 🚀 **Multi-Tier Hybrid OCR**: Combines the structural strengths of Baidu PaddleOCR (DBNet line detector) with Microsoft TrOCR (ViT encoder + RoBERTa decoder) for cursive character refinement.
* 🔍 **Adaptive Image Normalization**: Contrast Limited Adaptive Histogram Equalization (CLAHE) boosts faint pencil/pen strokes while suppressing background notebook textures.
* 📐 **Mathematical Geometric Line Clustering**: Computes bounding box vertical overlap ratios ($\ge 45\%$) to reconstruct reading order without line scrambling.
* 🧠 **LLM Semantic Post-Correction**: Local Ollama model (`qwen2.5`) repairs misread characters, inserts missing punctuation, and restores sentence grammar using domain context.
* 🛡️ **Anti-Hallucination Guardrails**: Strict 7-rule prompt design prevents the LLM from inventing facts, summarizing answers, or altering authentic student submissions.
* ⚡ **PyTorch Batch Inference & Auto-CUDA**: Groups handwriting crops into parallel tensor batches; automatically utilizes NVIDIA GPUs (e.g., RTX 3060 / 3070 / 4060) for near-instant inference.
* 📦 **Offline Fallback Architecture**: Automatic seamless fallback to `symspellpy` compound spelling dictionary lookups if the local LLM is offline.
* 💻 **Full-Stack Next.js 16 Web Dashboard**: Interactive web interface featuring side-by-side original page vs. extracted result, drag-and-drop multi-page PDF uploads, real-time progress bars, and confidence metrics.
* 🖱️ **Zero-Error 1-Click Launchers**: Double-clickable `setup.bat` and `run.bat` scripts for Windows (plus `setup.sh` and `run.sh` for Linux/macOS) for automated setup.

---

## 🏗️ System Architecture

```
                                  +------------------------------------+
                                  |   Uploaded PDF / Scanned Image     |
                                  +------------------------------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |   1. PDF Rasterization (PyMuPDF)   |
                                  |      - High-DPI page extraction    |
                                  +------------------------------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |   2. Adaptive Vision Preprocessing |
                                  |      - Dynamic Upscaling (>=2200px)|
                                  |      - Bilateral Noise Filtering   |
                                  |      - CLAHE Contrast Enhancement  |
                                  |      - Unsharp Stroke Masking      |
                                  +------------------------------------+
                                                     |
                                                     v
                         +---------------------------+---------------------------+
                         |                                                       |
                         v                                                       v
         +--------------------------------+                     +--------------------------------+
         |  PaddleOCR DBNet Detection     |                     |  PaddleOCR SVTR Recognition    |
         |  - Sensitive Threshold (0.10)  |                     |  - Full-width line recognition |
         |  - Box Boundary Expansion (2.0)|                     +--------------------------------+
         +--------------------------------+                                      |
                         |                                                       |
                         +---------------------------+---------------------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |  Short/Cursive Crop Filtering      |
                                  |  - Aspect ratio <= 4.5             |
                                  |  - Confidence < 0.88               |
                                  +------------------------------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |  Microsoft TrOCR (ViT + RoBERTa)   |
                                  |  - Parallel Batch CUDA Inference   |
                                  |  - Greedy Beam Decoding            |
                                  +------------------------------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |  3. Geometric Line Clustering      |
                                  |     Overlap_Y / Min(H1, H2) >= 0.45|
                                  |     Left-to-Right Word Assembly    |
                                  +------------------------------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |  4. Semantic Post-Correction       |
                                  |     Ollama (Qwen 2.5) 7 Rules      |
                                  |     (Fallback: SymSpell Compound)  |
                                  +------------------------------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |  5. Question / Answer Parser       |
                                  |     - Regex: Q# 01, Q# 02, etc.    |
                                  |     - Multi-line Spacing Formatter |
                                  +------------------------------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |  Next.js 16 Web Dashboard (React)  |
                                  |  - Side-by-Side Original vs Text   |
                                  |  - Interactive Confidence Badges   |
                                  +------------------------------------+
```

---

## 🔬 Deep-Dive: How the Pipeline Works

### Stage 1: Adaptive Preprocessing & Normalization
*File: [`ocr/image_processor.py`](file:///d:/dekstop%20win%2010/internship%20dekstop/deskstop/FYP/ai-ocr-system/ai-ocr-backend/ocr/image_processor.py)*

Handwritten documents scanned from mobile apps (CamScanner) or cameras suffer from uneven illumination, shadows, and low resolution.
1. **Dynamic Scaling**: Detects if $\max(H, W) < 2200\text{ px}$. If so, resizes using bicubic interpolation (`cv2.INTER_CUBIC`). This ensures that character stroke heights are at least $35\text{–}50\text{ px}$, preventing downsampling loss in DBNet's receptive field.
2. **Bilateral Filtering**: Unlike standard Gaussian blurs, `cv2.bilateralFilter(d=5, sigmaColor=40, sigmaSpace=40)` smooths paper grain and background artifacts while preserving character stroke edges.
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: Standard global histogram equalization blows out white background paper and turns ruled notebook lines pitch black. CLAHE computes equalization in small $8 \times 8$ local tiles with a clip limit of $2.2$, boosting faint pencil strokes naturally.
4. **Unsharp Masking**: Blends the equalized image with an inverted Gaussian blur ($1.25 \times I - 0.25 \times G$) to sharpen ink boundaries.

---

### Stage 2: Hybrid OCR Engine (PaddleOCR + TrOCR)
*File: [`ocr/handwriting_ocr.py`](file:///d:/dekstop%20win%2010/internship%20dekstop/deskstop/FYP/ai-ocr-system/ai-ocr-backend/ocr/handwriting_ocr.py)*

Pure Vision Transformer models (such as `microsoft/trocr-base-handwritten`) downsample all input crops to $384 \times 384\text{ px}$. When passed an entire line of text spanning $2400\text{ px}$ horizontally, the words are compressed into illegible artifacts. Conversely, standard CRNN models struggle with connected cursive handwriting.

Our hybrid engine solves this:
1. **Detection**: Configured with handwriting-specific DBNet thresholds:
   * `det_db_thresh = 0.10` (Captures faint, light pen lines).
   * `det_db_box_thresh = 0.20` (Retains candidate text boxes with low contrast).
   * `det_db_unclip_ratio = 2.0` (Expands box borders so letter ascenders/descenders like `g`, `y`, `f`, `t` are not clipped).
2. **Line Recognition**: PaddleOCR's SVTR sequence model recognizes wide text lines across arbitrary aspect ratios without squishing.
3. **Cursive Refinement via TrOCR**: Any text box with an aspect ratio $\le 4.5$, width $\ge 25\text{ px}$, and confidence $< 0.88$ is cropped with padding and queued for TrOCR.
4. **CUDA Batch Tensor Processing**: Instead of evaluating crops sequentially in a Python loop (which takes minutes on CPU), crops are batched into chunks of 16 and passed to TrOCR simultaneously (`model.generate(batch_tensors, num_beams=1)`), achieving **30x speedups**.

---

### Stage 3: Geometric Line Clustering & Reading Order
*File: [`ocr/handwriting_ocr.py`](file:///d:/dekstop%20win%2010/internship%20dekstop/deskstop/FYP/ai-ocr-system/ai-ocr-backend/ocr/handwriting_ocr.py)*

Handwritten lines drift diagonally, and students write in multi-segment lines. Sorting bounding boxes purely by centroid $Y$-coordinates frequently scrambles words from line 2 into line 1.

Our system uses **Geometric Vertical Bounding Box Overlap**:
$$\text{Overlap}_Y = \max(0, \min(y_1 + h_1, y_2 + h_2) - \max(y_1, y_2))$$
$$\text{Ratio} = \frac{\text{Overlap}_Y}{\min(h_1, h_2)}$$

* If $\text{Ratio} \ge 0.45$, the two bounding boxes share at least 45% of their vertical height and belong to the same logical line.
* Bounding boxes within each line cluster are then sorted left-to-right by $X$-coordinate:
  $$\text{line.sort}(\text{key}=\lambda\text{ box}: \text{box}[0])$$

---

### Stage 4: Semantic LLM Post-Correction (Ollama)
*File: [`ocr/text_cleaner.py`](file:///d:/dekstop%20win%2010/internship%20dekstop/deskstop/FYP/ai-ocr-system/ai-ocr-backend/ocr/text_cleaner.py)*

The raw OCR output is piped directly into a local Large Language Model via Ollama (`qwen2.5:1.5b` or `qwen2.5:7b`).

```
Raw Vision OCR:
"laxge Longuage ModcsLLMs ave AI models . trained on exergic amounb of kxt..."
                                   ↓
Contextual LLM Post-Correction:
"Large Language Models (LLMs) are AI models trained on large amounts of text..."
```

#### **The 7 Strict Rules in Prompt Engineering:**
1. **Fix spelling and obvious OCR mistakes**: Corrects phonetic and visual character confusions based on domain context (`Jorge Languaqe` $\rightarrow$ `Large Language`, `teild` $\rightarrow$ `field`).
2. **Fix missing spaces, broken words, and punctuation**: Inserts spaces between merged words and restores commas, periods, and sentence endings.
3. **Correct words using surrounding context**: Disambiguates ambiguous words using the subject of the document.
4. **Preserve technical terms, numbers, formulas, and code**: Protects programming syntax (`int`, `float`, `// octal integer`, `C++`, `LLMs`).
5. **Anti-Hallucination Guardrail**: Strictly forbids inventing facts or adding external knowledge not supported by the handwriting.
6. **No Summarization**: Prevents the model from paraphrasing or shortening student answers.
7. **Line & Structure Preservation**: Preserves question headers (`Q# 01:`), numbered rules (`Rule 1:`), and bullet points on separate lines.

#### **Offline Fallback:**
If Ollama is not installed or unreachable, the system automatically falls back to `symspellpy` compound spell checking with frequency dictionaries (`frequency_dictionary_en_82_765.txt`), ensuring the system **never crashes**.

---

### Stage 5: Question/Answer Parsing & Disambiguation
*File: [`app.py`](file:///d:/dekstop%20win%2010/internship%20dekstop/deskstop/FYP/ai-ocr-system/ai-ocr-backend/app.py)*

The cleaned lines are parsed by an adaptive regex engine:
* Matches question markers: `Q# 01`, `Q#02:-`, `Question 1:`, `Q1:`, `QB 01`.
* Formats answer bodies using `format_answer_lines()`, maintaining distinct linebreaks (`\n`) for bullet points and code comments.
* Calculates confidence per question by averaging word confidences from the vision engine.

---

## 📁 Repository Directory Structure

```text
ai-ocr-system/
├── .gitignore                     # Ignores venv, node_modules, .next, weights, and temp uploads
├── README.md                      # Comprehensive project documentation
├── setup.bat                      # 1-Click automated installer for Windows
├── run.bat                        # 1-Click double-clickable launcher for Windows
├── setup.sh                       # Automated installer for Linux / macOS
├── run.sh                         # Multi-service launcher for Linux / macOS
│
├── ai-ocr-backend/                # FastAPI Python Backend
│   ├── app.py                     # Main server: routes (/upload, /history), CORS, orchestration
│   ├── database.py                # SQLAlchemy SQLite ORM models (Uploads, ExtractedQuestions)
│   ├── download_trocr.py          # Utility script to pre-download TrOCR weights locally
│   ├── requirements.txt           # Strictly pinned Python dependencies (PaddleOCR 2.9, NumPy<2)
│   ├── .env.example               # Template environment configuration
│   ├── .env                       # Active runtime configuration (Ollama URL, model name)
│   │
│   ├── ocr/                       # Deep Learning & OCR Pipeline Package
│   │   ├── __init__.py            # Package initializer
│   │   ├── pdf_converter.py       # High-DPI PDF page rasterization using PyMuPDF (fitz)
│   │   ├── image_processor.py     # CLAHE, dynamic scaling, bilateral filtering, unsharp masking
│   │   ├── handwriting_ocr.py     # PaddleOCR DBNet + Microsoft TrOCR ViT + CUDA batch inference
│   │   └── text_cleaner.py        # Ollama LLM contextual post-correction + SymSpell fallback
│   │
│   ├── trocr-base-handwritten/    # (Optional) Local offline model folder (~1.33 GB)
│   ├── uploads/                   # Temporary storage for uploaded PDF files
│   ├── processed/                 # Rendered PNG page images served via /images/
│   └── results/                   # JSON exports of processed document results
│
└── ai-ocr-frontend/               # Next.js 16 + React 19 Frontend
    ├── package.json               # Node.js production & development dependencies
    ├── next.config.ts             # Next.js runtime configuration
    ├── tsconfig.json              # TypeScript configuration
    ├── postcss.config.mjs         # Tailwind CSS PostCSS configuration
    ├── public/                    # Static UI icons & SVG assets
    │
    └── src/
        └── app/
            ├── layout.tsx         # Global layout & metadata
            ├── globals.css        # Tailwind CSS styles & typography
            └── page.tsx           # Interactive OCR dashboard, dropzone, Q&A cards, history
```

---

## 🛠️ Tech Stack & Exact Library Versions

### **Backend Python Environment (`ai-ocr-backend`)**
| Library | Exact Version | Purpose |
|---|---|---|
| `python` | `3.11.x` | Core Python runtime environment |
| `paddleocr` | `2.9.1` | DBNet text line bounding box detection & SVTR recognition |
| `paddlepaddle` | `2.6.2` | High-performance C++ deep learning backend for PaddleOCR |
| `torch` | `2.13.0` (or `cu121`) | PyTorch tensor compute engine for TrOCR Vision Transformer |
| `torchvision` | `0.28.0` | Computer vision tensor transforms |
| `transformers` | `5.15.0` | Hugging Face library running `microsoft/trocr-base-handwritten` |
| `opencv-python` | `4.10.0.84` | Image preprocessing (CLAHE, bilateral filter, unsharp mask) |
| `numpy` | `1.26.4` (`<2.0.0`) | Fast array manipulation (strictly pinned `<2` for C-ABI safety) |
| `PyMuPDF` (`fitz`) | `1.28.2` | High-resolution PDF page rendering |
| `pillow` | `12.3.0` | PIL image loader for transformer inputs |
| `symspellpy` | `6.10.0` | High-speed compound spelling dictionary (offline fallback) |
| `fastapi` | `0.142.2` | Asynchronous REST API web framework |
| `uvicorn` | `0.54.0` | Production ASGI web server |
| `SQLAlchemy` | `2.0.49` | SQLite ORM database for upload history and results |
| `python-dotenv` | `1.2.4` | Environment variable loader |

### **Frontend Environment (`ai-ocr-frontend`)**
| Package | Version | Purpose |
|---|---|---|
| `next` | `16.2.6` | React production framework with App Router |
| `react` | `19.2.4` | Modern UI component rendering |
| `tailwindcss` | `^4.0.0` | Utility-first CSS styling engine |
| `axios` | `^1.16.0` | HTTP client for backend API communication |
| `lucide-react` | `^1.14.0` | Modern SVG icon set |
| `react-dropzone` | `^15.0.0` | Drag-and-drop file upload component |
| `framer-motion` | `^12.38.0` | Smooth UI animations and micro-interactions |

### **AI Models**
* **TrOCR Base Handwritten**: `microsoft/trocr-base-handwritten` (334M parameters, ViT-B encoder + RoBERTa decoder).
* **PaddleOCR English**: `ch_PP-OCRv4` / `en_PP-OCRv3` (Baidu DBNet + SVTR).
* **Ollama Model**: `qwen2.5:1.5b` (Recommended default) or `qwen2.5:7b` (For high-spec GPUs).

---

## ⚡ Installation & Quick Start

### 1-Click Setup (Windows - Recommended)
1. Clone the repository:
   ```cmd
   git clone https://github.com/AryanAfzal/OCR_SYSTEM.git
   cd OCR_SYSTEM
   ```
2. **Double-click `setup.bat`**:
   * Automatically detects Python 3.11/3.10 and Node.js.
   * Creates the virtual environment and installs all pinned dependencies.
   * Generates `.env`, builds required directories, and pulls the Ollama model.
3. **Double-click `run.bat`**:
   * Launches both the backend and frontend in separate windows.
   * Automatically opens **[http://localhost:3000](http://localhost:3000)** in your browser!

---

### 1-Click Setup (Linux / macOS)
```bash
git clone https://github.com/AryanAfzal/OCR_SYSTEM.git
cd OCR_SYSTEM
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

---

### Manual Step-by-Step Setup

#### **1. Backend Setup**
```bash
cd ai-ocr-backend

# Create virtual environment with Python 3.11
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# (Optional) For NVIDIA CUDA GPU acceleration:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Setup configuration
cp .env.example .env

# (Optional) Pre-download TrOCR model locally for instant offline boots:
python download_trocr.py

# Start the FastAPI server
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

#### **2. Ollama AI Setup**
```bash
# Install Ollama from https://ollama.ai, then run:
ollama pull qwen2.5:1.5b
ollama run qwen2.5:1.5b
```

#### **3. Frontend Setup**
```bash
cd ai-ocr-frontend
npm install
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)**.

---

## 🏎️ Hardware Acceleration & Performance Benchmarks

| Hardware | Document Type | Page Count | Total Processing Time | Tokens / Sec |
|---|---|---|---|---|
| **Intel Core i5 / i7 (CPU-Only)** | 1 Page Exam Sheet | 1 Page | ~4.5 seconds | ~12 t/s |
| **Intel Core i5 / i7 (CPU-Only)** | Multi-question Booklet | 4 Pages | ~16 seconds | ~12 t/s |
| **NVIDIA RTX 3060 / 3070 (CUDA)** | 1 Page Exam Sheet | 1 Page | **~0.9 seconds** | ~85 t/s |
| **NVIDIA RTX 3070 (CUDA Batching)**| 30 Exam Papers Batch | 105 Pages | **~1.8 minutes** | ~90 t/s |

---

## 🔌 REST API Reference

### 1. `POST /upload`
Uploads and processes a multi-page handwritten PDF with live server-sent progress.

* **URL**: `http://127.0.0.1:8000/upload`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Body**: `file` (Binary PDF document)

#### **Success Response (200 OK)**:
```json
{
  "upload_id": 1,
  "total_pages": 1,
  "images": [
    "http://127.0.0.1:8000/images/page_1.png"
  ],
  "questions": [
    {
      "question": "Q# 01",
      "answer": "Large Language Models (LLMs) are AI models trained on large amounts of text to understand and generate human-like language. They can perform tasks such as answering questions, summarizing, translating, and coding. Examples include GPT, Llama, Claude, and Gemini.",
      "confidence": 0.98
    }
  ],
  "raw_text": [
    {
      "text": "Q# 01",
      "confidence": 0.98
    },
    {
      "text": "Large Language Models (LLMs) are AI models trained on large amounts of text...",
      "confidence": 0.98
    }
  ]
}
```

---

### 2. `GET /history`
Retrieves past processed documents.

* **URL**: `http://127.0.0.1:8000/history`
* **Method**: `GET`
* **Response**: List of past document summaries with IDs, filenames, page counts, and timestamps.

---

### 3. `GET /history/{upload_id}`
Retrieves extracted questions and page images for a previously processed document.

* **URL**: `http://127.0.0.1:8000/history/1`
* **Method**: `GET`

---

## 🔧 Troubleshooting & Common Pitfalls

### 1. `RuntimeError: module compiled against ABI version 0x1000009 but this version of numpy is 0x2000000`
* **Cause**: NumPy 2.x was installed globally. OpenCV, PyTorch, and PaddlePaddle require NumPy 1.x ABI.
* **Fix**: Ensure your environment has `numpy<2` installed:
  ```powershell
  python -m pip install "numpy<2" "opencv-python<=4.10.0.84"
  ```

### 2. `ValueError: Unknown argument: use_gpu` / `OneDnnContext Error`
* **Cause**: Installing unpinned `paddleocr` installs PaddleOCR 3.x (PaddleX) which removed the `use_gpu` parameter.
* **Fix**: Strictly pin PaddleOCR in `requirements.txt`:
  ```powershell
  pip install paddleocr==2.9.1 paddlepaddle==2.6.2
  ```

### 3. Python 3.14 / 3.12 Default Conflicts on Windows
* **Cause**: Deep learning libraries (PaddlePaddle, PyTorch) do not yet have pre-built Windows wheels for Python 3.14.
* **Fix**: Install Python 3.11. Our `setup.bat` script automatically looks for `py -3.11` first.

### 4. Application Control Policy Blocks `uvicorn.exe`
* **Cause**: Windows security policies restrict executing third-party binaries directly from `venv\Scripts\`.
* **Fix**: Always start Uvicorn via Python module syntax:
  ```powershell
  python -m uvicorn app:app --reload --port 8000
  ```

---

## 🚀 Future Roadmap & FYP Extensions

- [ ] **Automated AI Grading Engine**: Integrate rubric-based grading where the system compares student answers against a marking scheme and assigns scores with explanations.
- [ ] **Cross-Student Plagiarism & Collusion Detector**: Vectorize answers using sentence transformers to identify cheating or copied text across exam booklets.
- [ ] **Math Formula & Chemistry LaTeX Converter**: Add mathematical handwriting recognition (pix2tex) for equations.
- [ ] **Export to Word / Excel / LMS**: One-click export to `.docx`, `.xlsx`, or direct grade sync with Canvas, Google Classroom, and Moodle.

---

## 📄 License & Acknowledgements

* **License**: Distributed under the **MIT License**. See `LICENSE` for details.
* **TrOCR**: Microsoft Research ([TrOCR on Hugging Face](https://huggingface.co/microsoft/trocr-base-handwritten)).
* **PaddleOCR**: Baidu PaddlePaddle Open Source Team ([PaddleOCR Repository](https://github.com/PaddlePaddle/PaddleOCR)).
* **Ollama**: Community-driven local LLM runtime ([Ollama.ai](https://ollama.ai)).
* **Author**: Aryan Afzal & FYP Engineering Team.
