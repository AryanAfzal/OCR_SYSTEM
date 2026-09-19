import fitz
import os

def convert_pdf_to_images(pdf_path, output_folder="processed"):
    os.makedirs(output_folder, exist_ok=True)

    pdf = fitz.open(pdf_path)
    image_paths = []

    for page_number in range(len(pdf)):
        page = pdf[page_number]

        # 4.0 zoom = 288 DPI — higher resolution improves PaddleOCR accuracy on handwriting
        zoom = 4.0
        matrix = fitz.Matrix(zoom, zoom)
        image = page.get_pixmap(matrix=matrix)

        image_filename = f"page_{page_number + 1}.png"
        image_path = os.path.join(output_folder, image_filename)
        image.save(image_path)
        image_paths.append(image_path)

    pdf.close()
    return image_paths