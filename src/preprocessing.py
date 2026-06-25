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
