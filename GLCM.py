import numpy as np
from skimage.feature import graycomatrix, graycoprops
# from skimage import io, color, img_as_ubyte

# GLCM properties
def contrast_feature(matrix_coocurrence):
    contrast = graycoprops(matrix_coocurrence, 'contrast')
    return contrast

def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = graycoprops(matrix_coocurrence, 'dissimilarity')
    return dissimilarity

def homogeneity_feature(matrix_coocurrence):
    homogeneity = graycoprops(matrix_coocurrence, 'homogeneity')
    return homogeneity

def energy_feature(matrix_coocurrence):
    energy = graycoprops(matrix_coocurrence, 'energy')
    return energy

def correlation_feature(matrix_coocurrence):
    correlation = graycoprops(matrix_coocurrence, 'correlation')
    return correlation


def entropy_feature(matrix_coocurrence):
    entropy = graycoprops(matrix_coocurrence, 'entropy')
    return entropy