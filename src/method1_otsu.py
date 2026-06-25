import cv2
import numpy as np


def segment_otsu(gray):
  
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def morphological_cleanup(binary):
  
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    cleaned = cv2.dilate(eroded, kernel, iterations=1)
    return cleaned
