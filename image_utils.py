from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import median
from skimage.morphology import ball

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    return image

def edge_detection(image):
    grayscale_image = np.mean(image, axis=2)
    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    kernelY = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])
    edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
