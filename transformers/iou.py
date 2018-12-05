import numpy as np


def diag_iou(boxes1, boxes2):
    x11, y11, x12, y12 = [boxes1[:, i] for i in range(4)]
    x21, y21, x22, y22 = [boxes2[:, i] for i in range(4)]

    x1 = np.maximum(x11, x21)
    y1 = np.maximum(y11, y21)
    x2 = np.minimum(x12, x22)
    y2 = np.minimum(y12, y22)

    intersection = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    return intersection / (box1_area + box2_area - intersection + 1e-8)


def full_iou(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    x1 = np.maximum(x11, np.transpose(x21))
    y1 = np.maximum(y11, np.transpose(y21))
    x2 = np.minimum(x12, np.transpose(x22))
    y2 = np.minimum(y12, np.transpose(y22))

    intersection = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    iou = intersection / (box1_area + np.transpose(box2_area) - intersection)
    return iou
