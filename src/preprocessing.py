import cv2


def load_image(path):
   
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    print(f"Image loaded: {path}")
    return img


def get_green_channel(img):


    green_channel = img[:, :, 1]
    print("Green channel extracted")
    return green_channel


def apply_clahe(gray, clip=2.0, tile=8):
    
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def apply_gaussian(img, ksize=5):
   
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def preprocess(path):
    
    original_bgr = load_image(path)
    green = get_green_channel(original_bgr)
    clahe_applied = apply_clahe(green)
    preprocessed_gray = apply_gaussian(clahe_applied)
    return original_bgr, preprocessed_gray


if __name__ == "__main__":
    import sys
    orig, processed = preprocess(sys.argv[1])
    print(f"Original shape: {orig.shape}")
    print(f"Processed shape: {processed.shape}")
