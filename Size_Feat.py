import cv2
import numpy as np


def extract_image_size_features(Image):
    Size_Feat = []
    for s in range(len(Image)):
        img = Image[s]
        height, width = img.shape
        area = height * width

        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour's bounding box
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bounding_box_area = w * h
            bounding_box_aspect_ratio = w / h if h != 0 else 0
        else:
            bounding_box_area = 0
            bounding_box_aspect_ratio = 0

        Feat = [height, width, area, bounding_box_area, bounding_box_aspect_ratio]
        Size_Feat.append(Feat)

    return np.asarray(Size_Feat)
