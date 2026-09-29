import cv2
import numpy as np
from skimage.measure import label, regionprops


def shape_features_extraction(image_path):
    image = image_path
    Feats = []
    for j in range(len(image)):
        Img = image[j]
        _, binary_image = cv2.threshold(Img, 127, 255, cv2.THRESH_BINARY)
        labeled_image = label(binary_image)
        # Get the properties of the labeled regions
        regions = regionprops(labeled_image)
        for region in regions:
            # Extract shape features
            area = region.area
            perimeter = region.perimeter
            centroid = region.centroid
            bounding_box = region.bbox
            aspect_ratio = (bounding_box[3] - bounding_box[1]) / (bounding_box[2] - bounding_box[0])
            extent = region.extent
            solidity = region.solidity
            feat = [area, centroid[0], bounding_box[0], aspect_ratio, extent, solidity]
            Feats.append(feat)
    return np.asarray(Feats)


