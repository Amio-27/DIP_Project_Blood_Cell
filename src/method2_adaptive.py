import cv2
import numpy as np


def segment_adaptive(gray):
  
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    return binary


def apply_morphology(binary):
  
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing


def run(img_path):
 
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import preprocess

    orig_bgr, gray = preprocess(img_path)
    binary = segment_adaptive(gray)
    cleaned = apply_morphology(binary)
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/method2_adaptive_result.jpg", cleaned)
    return cleaned


if __name__ == "__main__":
    import sys
    mask = run(sys.argv[1])
    print(f"Mask shape: {mask.shape}")
    print(f"Unique values: {list(set(mask.flatten()))}")
