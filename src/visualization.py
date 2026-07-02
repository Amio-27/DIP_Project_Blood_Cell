import os
import sys
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def plot_preprocessing(img_path):
    """Plot original, green, CLAHE, and gaussian steps."""
    from preprocessing import preprocess

    orig_bgr, processed = preprocess(img_path)
    green = orig_bgr[:, :, 1]
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    images = [orig_rgb, green, processed, processed]
    titles = ["Original", "Green", "CLAHE", "Gaussian"]

    for ax, img, title in zip(axes, images, titles):
        if title == "Original":
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/preprocessing_steps.jpg")
    plt.close()


def plot_all_methods(img_path):
    """Plot Otsu, Adaptive, HSV, Watershed results."""
    from method1_otsu import run as run_otsu
    from method2_adaptive import run as run_adaptive
    from method3_hsv import run as run_hsv

    try:
        from method4_watershed import run as run_watershed
    except ImportError:
        run_watershed = None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    results = [
        run_otsu(img_path),
        run_adaptive(img_path),
        run_hsv(img_path),
        run_watershed(img_path) if run_watershed else None,
    ]
    titles = ["Otsu", "Adaptive", "HSV", "Watershed"]

    for ax, result, title in zip(axes.flat, results, titles):
        if result is not None:
            ax.imshow(result, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/all_methods_comparison.jpg")
    plt.close()
