import cv2
import numpy as np
import os

def segment_rbc(img_bgr):

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 50, 50], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 50, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    mask2 = cv2.inRange(img_hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    print("RBC mask created")
    return mask


def segment_wbc(img_bgr):
    """
    Segment White Blood Cells using HSV color thresholding.
    WBC appears purple/dark due to nucleus staining.
    HSV is used instead of BGR because it separates
    color (hue) from brightness, making thresholding
    more robust to lighting changes.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([125, 30, 30], dtype=np.uint8)
    upper = np.array([160, 255, 200], dtype=np.uint8)

    mask = cv2.inRange(img_hsv, lower, upper)

    print("WBC mask created")
    return mask


def combine_masks(rbc_mask, wbc_mask):

    combined = np.zeros_like(rbc_mask)
    combined[rbc_mask > 0] = 100
    combined[wbc_mask > 0] = 200

    print("Masks combined: 0=background, 100=RBC, 200=WBC")
    return combined


def run(img_path):

    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    rbc_mask = segment_rbc(img_bgr)
    wbc_mask = segment_wbc(img_bgr)
    mask = combine_masks(rbc_mask, wbc_mask)

    overlay = img_bgr.copy()
    overlay[mask == 100] = [0, 0, 255]
    overlay[mask == 200] = [255, 0, 0]

    os.makedirs("results", exist_ok=True)
    saved = cv2.imwrite("results/method3_hsv_result.jpg", overlay)
    if not saved:
        print("Warning: Failed to save result image.")

    print("Method 3 complete. Result saved.")
    return mask


if __name__ == "__main__":
    import sys
    mask = run(sys.argv[1])
    print(f"Output mask shape: {mask.shape}")
    unique = [int(v) for v in set(mask.flatten())]
    print(f"Unique values (0=bg, 100=RBC, 200=WBC): {unique}")