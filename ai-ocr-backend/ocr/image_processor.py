import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return image_path

    h, w = image.shape[:2]

    # 1. Dynamic scaling: Ensure image has at least ~2200px on the longest dimension
    # This prevents DBNet detector from missing small handwriting lines in low-res scans
    max_dim = max(h, w)
    if max_dim < 2200:
        scale = 2200.0 / float(max_dim)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 3. Bilateral filter: smooths background paper texture while keeping stroke edges sharp
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)

    # 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # Enhances faint/light pen strokes across varied document lighting
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Gentle unsharp masking for crisp character strokes
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(enhanced, 1.25, gaussian, -0.25, 0)

    cv2.imwrite(image_path, sharpened)
    return image_path


def preprocess_all_images(image_paths):
    cleaned = []
    for path in image_paths:
        cleaned.append(preprocess_image(path))
    return cleaned


