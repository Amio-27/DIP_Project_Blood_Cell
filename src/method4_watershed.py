import cv2
import numpy as np


def get_sure_foreground(binary):
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    return sure_fg.astype(np.uint8)


def get_markers(sure_fg, binary):
    sure_bg = cv2.dilate(binary, np.ones((3, 3)), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    return markers
