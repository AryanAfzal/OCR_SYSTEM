import os
import sys
from transformers import ViTImageProcessor, RobertaTokenizer, VisionEncoderDecoderModel

MODEL_NAME = "microsoft/trocr-base-handwritten"
TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "trocr-base-handwritten"))

def main():
    print("=" * 60)
    print(f"Downloading TrOCR Model ({MODEL_NAME}) for Offline Use...")
    print("=" * 60)
    print(f"Target directory: {TARGET_DIR}")
    print()

    os.makedirs(TARGET_DIR, exist_ok=True)

    print("[1/3] Downloading Image Processor...")
    image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    image_processor.save_pretrained(TARGET_DIR)

    print("[2/3] Downloading Tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(TARGET_DIR)

    print("[3/3] Downloading VisionEncoderDecoder Model (~1.33 GB)...")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.save_pretrained(TARGET_DIR)

    print()
    print("=" * 60)
    print("[SUCCESS] TrOCR Model downloaded and saved to 'trocr-base-handwritten'!")
    print("The backend will now boot in 100% offline mode instantly.")
    print("=" * 60)

if __name__ == "__main__":
    main()
