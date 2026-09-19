import re
import os
import json
import urllib.request
from symspellpy import SymSpell
import pkg_resources

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))

# Initialize SymSpell for fast and accurate spelling correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
try:
    dict_path = pkg_resources.resource_filename('symspellpy', 'frequency_dictionary_en_82_765.txt')
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    
    bigram_path = pkg_resources.resource_filename('symspellpy', 'frequency_bigramdictionary_en_243_342.txt')
    if os.path.exists(bigram_path):
        sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
except Exception as e:
    print(f"SymSpell dictionary init: {e}")

# Generic OCR confusion patterns for handwritten English
OCR_REPLACEMENTS = [
    (r'\b0([a-zA-Z]{2,})\b', r'o\1'),      # 0ver -> over
    (r'\b1([a-zA-Z]{2,})\b', r'l\1'),      # 1azy -> lazy
    (r'^\s*([QqGgBb#\s]*0?(\d+))[:.-]?\s*$', r'Q# \2:'), # Question numbering header normalization
    (r'\b([Qq][#\s]*0?(\d+))\b', r'Q# \2:'),
]

# Watermarks and noise to ignore
IGNORED_PATTERNS = [
    r'^.*camscanner.*$',
    r'^cs\s*camscanner$',
    r'^camscanner$',
    r'^scanned\s+with.*$',
    r'^scanned\s+by.*$',
    r'^\*?\*?corrected text:?\*?\*?$',
    r'^here is the corrected.*$',
    r'^[\W_]+$', # only punctuation or symbols
    r'^[\d\s.,;:\-_=+/\\|~*^]+$', # only digits & punctuation without real words
]


def is_watermark_or_noise(text: str) -> bool:
    cleaned = text.strip().lower()
    for pattern in IGNORED_PATTERNS:
        if re.match(pattern, cleaned):
            return True
    return False


def correct_with_ollama(full_text: str) -> str:
    """
    Sends raw OCR document text to local Ollama (qwen2.5:1.5b) for high-accuracy
    contextual error restoration and formatting.
    """
    prompt = (
        "You are an OCR post-processing system.\n\n"
        "The following text was extracted from a handwritten document.\n"
        "Correct OCR errors while preserving the original meaning.\n\n"
        "Rules:\n"
        "1. Fix spelling and obvious OCR mistakes (e.g., 'Jorge Languaqe Models' -> 'Large Language Models', 'feild' -> 'field', 'requlaly' -> 'regularly').\n"
        "2. Fix missing spaces, broken words, and punctuation.\n"
        "3. Correct words using the surrounding context.\n"
        "4. Preserve technical terms, numbers, formulas, and code.\n"
        "5. Do not add information that is not supported by the OCR text.\n"
        "6. Do not summarize or rewrite unnecessarily.\n"
        "7. Preserve question headers (e.g., 'Q# 01:') and list formatting on separate lines.\n"
        "8. Return ONLY the corrected text directly without preamble or commentary.\n\n"
        "OCR TEXT:\n"
        f"{full_text}"
    )

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        corrected = data.get("response", "").strip()
        
        # Clean intro lines if any
        lines = [l.strip() for l in corrected.split("\n") if l.strip()]
        clean_lines = [l for l in lines if not is_watermark_or_noise(l)]
        return "\n".join(clean_lines)
    return full_text


def correct_spelling_symspell(text: str) -> str:
    """
    Fast offline compound spell checking using SymSpell and regex rules.
    """
    if not text:
        return text

    # Preserve question headers like "Q1:", "Q# 01:"
    header_match = re.match(r'^(([Qq]\s*#?\s*\d+|Question\s*\d+)[:.]?\s*)(.*)$', text, re.IGNORECASE)
    if header_match:
        prefix = header_match.group(1)
        body = header_match.group(3)
    else:
        prefix = ""
        body = text

    # Apply known OCR pattern replacements
    for pattern, repl in OCR_REPLACEMENTS:
        body = re.sub(pattern, repl, body, flags=re.IGNORECASE)

    # Use SymSpell compound lookup
    try:
        suggestions = sym_spell.lookup_compound(body, max_edit_distance=2)
        if suggestions and suggestions[0].term:
            corrected = suggestions[0].term
            if body and body[0].isupper() and len(corrected) > 0:
                corrected = corrected[0].upper() + corrected[1:]
            body = corrected
    except Exception:
        pass

    return f"{prefix}{body}".strip()


def clean_results(ocr_results: list) -> list:
    """
    Cleans, spell-corrects, and restores full OCR results using Ollama AI when available,
    with automatic fallback to SymSpell.
    """
    # 1. Filter raw OCR lines and pre-normalize OCR confusables
    valid_lines = []
    for item in ocr_results:
        raw = item["text"].strip()
        if raw and not is_watermark_or_noise(raw):
            for pattern, repl in OCR_REPLACEMENTS:
                raw = re.sub(pattern, repl, raw, flags=re.IGNORECASE)
            raw = raw.strip()
            if raw and not is_watermark_or_noise(raw):
                valid_lines.append(raw)

    if not valid_lines:
        return []

    combined_text = "\n".join(valid_lines)

    # 2. Try Ollama AI post-correction first
    try:
        print(f"Applying AI Post-Correction with Ollama ({OLLAMA_MODEL})...")
        ai_corrected_text = correct_with_ollama(combined_text)
        corrected_lines = [line.strip() for line in ai_corrected_text.split("\n") if line.strip()]
        
        final_results = []
        for line in corrected_lines:
            if not is_watermark_or_noise(line):
                final_results.append({
                    "text": line,
                    "confidence": 0.98
                })
        if final_results:
            print(f"Ollama post-correction applied successfully ({len(final_results)} lines).")
            return final_results
    except Exception as e:
        print(f"Ollama AI correction unavailable ({e}). Falling back to SymSpell...")

    # 3. Fallback to SymSpell & Regex
    cleaned = []
    for item in ocr_results:
        raw_text = item["text"].strip()
        if not raw_text or is_watermark_or_noise(raw_text):
            continue

        for pattern, repl in OCR_REPLACEMENTS:
            raw_text = re.sub(pattern, repl, raw_text, flags=re.IGNORECASE)

        cleaned_text = correct_spelling_symspell(raw_text)
        if cleaned_text and not is_watermark_or_noise(cleaned_text):
            cleaned.append({
                "text": cleaned_text,
                "confidence": item.get("confidence", 0.90)
            })

    return cleaned


