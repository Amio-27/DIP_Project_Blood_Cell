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


def run(img_path):
    """Run Otsu segmentation pipeline on an image path."""
    import os
    from preprocessing import preprocess

    orig_bgr, gray = preprocess(img_path)
    binary = segment_otsu(gray)
    cleaned = morphological_cleanup(binary)
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/method1_otsu_result.jpg", cleaned)
    return cleaned


if __name__ == "__main__":
    import sys
    mask = run(sys.argv[1])
    print(f"Mask shape: {mask.shape}")
    print(f"Unique values: {list(set(mask.flatten()))}")
