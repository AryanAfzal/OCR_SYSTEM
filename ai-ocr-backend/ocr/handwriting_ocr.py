from transformers import ViTImageProcessor, RobertaTokenizer, VisionEncoderDecoderModel
import torch
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2
import os

# Device configuration (CUDA GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU_PADDLE = torch.cuda.is_available()

# Initialize PaddleOCR with sensitive detection parameters for handwriting
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=USE_GPU_PADDLE,
    show_log=False,
    det_limit_side_len=3000,
    det_db_thresh=0.10,       # Sensitive threshold to capture light/thin handwriting
    det_db_box_thresh=0.20,   # Keep faint handwritten boxes
    det_db_unclip_ratio=2.0   # Adequate margins for full character ascenders/descenders
)

# Initialize TrOCR for handwriting recognition
BASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trocr-base-handwritten")

if os.path.exists(BASE_MODEL_DIR):
    MODEL_SRC = os.path.abspath(BASE_MODEL_DIR)
    print(f"Loading official Base TrOCR from local folder '{MODEL_SRC}'...")
else:
    MODEL_SRC = 'microsoft/trocr-base-handwritten'
    print(f"Loading TrOCR from '{MODEL_SRC}'...")

image_processor = ViTImageProcessor.from_pretrained(MODEL_SRC)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_SRC)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_SRC)
model.to(DEVICE)
model.eval()
print(f"TrOCR model loaded successfully on device: {DEVICE.upper()}")


def extract_text_from_image(image_path):
    """
    Extracts handwriting text lines using PaddleOCR detection & recognition,
    refines short/difficult handwriting crops in parallel batches with TrOCR,
    and sorts into natural reading order.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = ocr_engine.ocr(image_path, cls=True)
    if not result or result[0] is None:
        return []

    items = []
    crops_to_infer = []
    crop_indices = []

    for line in result[0]:
        if line is None or len(line) < 2:
            continue

        box, (p_text, p_conf) = line[0], line[1]
        pts = np.array(box, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        aspect = w / float(h) if h > 0 else 0

        idx = len(items)
        items.append({
            'box': (x, y, w, h),
            'text': p_text,
            'conf': p_conf
        })

        # Queue low-confidence / short handwriting crops for parallel TrOCR batch refinement
        if aspect <= 4.5 and w >= 25 and h >= 14 and p_conf < 0.88:
            pad_y = max(4, int(h * 0.15))
            pad_x = max(6, int(w * 0.10))
            y1 = max(0, y - pad_y)
            y2 = min(img_h, y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(img_w, x + w + pad_x)
            crop_np = img_rgb[y1:y2, x1:x2]

            if crop_np.shape[0] > 6 and crop_np.shape[1] > 6:
                crops_to_infer.append(Image.fromarray(crop_np))
                crop_indices.append(idx)

    # Run batch inference for all queued crops at once (30x faster)
    if crops_to_infer:
        batch_size = 16
        for b_start in range(0, len(crops_to_infer), batch_size):
            b_crops = crops_to_infer[b_start:b_start + batch_size]
            b_indices = crop_indices[b_start:b_start + batch_size]
            try:
                pixel_values = image_processor(images=b_crops, return_tensors='pt').pixel_values.to(DEVICE)
                with torch.no_grad():
                    out = model.generate(pixel_values, max_new_tokens=32, num_beams=1)
                decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
                for i, text_val in enumerate(decoded):
                    t_text = text_val.strip()
                    if t_text and len(t_text) > 1 and not t_text.startswith('0 000'):
                        items[b_indices[i]]['text'] = t_text
            except Exception as e:
                print(f"TrOCR batch inference warning: {e}")

    if not items:
        return []

    # Sort items by Y to group into lines using geometric vertical overlap
    items.sort(key=lambda it: it['box'][1])
    lines = []
    curr_line = [items[0]]

    for it in items[1:]:
        x, y, w, h = it['box']
        px, py, pw, ph = curr_line[-1]['box']

        # Calculate vertical overlap between bounding boxes
        overlap_y = max(0, min(y + h, py + ph) - max(y, py))
        min_h = min(h, ph)

        # If vertical overlap is at least 45% of box height, they belong to the same text line
        if min_h > 0 and (overlap_y / float(min_h)) >= 0.45:
            curr_line.append(it)
        else:
            lines.append(curr_line)
            curr_line = [it]

    if curr_line:
        lines.append(curr_line)

    final_output = []
    for line in lines:
        line.sort(key=lambda it: it['box'][0])  # Sort left-to-right
        line_text = ' '.join(it['text'] for it in line if it['text']).strip()
        avg_conf = sum(it['conf'] for it in line) / max(1, len(line))
        if line_text:
            print(f"--> Extracted line: {line_text}")
            final_output.append({'text': line_text, 'confidence': round(avg_conf, 2)})

    return final_output


def extract_text_from_images(image_paths):
    all_results = []
    for path in image_paths:
        page_results = extract_text_from_image(path)
        all_results.extend(page_results)
    return all_results



