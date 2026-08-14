import cv2
import pytesseract
import numpy as np

def extract_text_from_image(file):
    # Convert file to OpenCV image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(img)
    return text
