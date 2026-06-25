# DIP Project: Blood Cell Segmentation

This project contains image-processing utilities for segmenting red blood
cells (RBCs) and white blood cells (WBCs) from blood-smear images.

## Current method

`src/method3_hsv.py` uses OpenCV's HSV color space:

- RBCs are detected from two red/pink hue ranges.
- WBCs are detected from a purple/dark hue range.

Each function accepts a BGR image and returns a binary mask.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
import cv2

from src.method3_hsv import segment_rbc, segment_wbc

image = cv2.imread("path/to/blood-smear-image.jpg")
rbc_mask = segment_rbc(image)
wbc_mask = segment_wbc(image)
```
