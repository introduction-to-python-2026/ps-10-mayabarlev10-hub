from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def load_image(path):
    img = Image.open(path)
    # המרה למערך נומפי
    image_array = np.array(img)
    return image_array # חשוב להחזיר את המערך!

   
def edge_detection(image_array):
    # 1. המרה לגווני אפור
    gray_image = np.mean(image_array, axis=2)
    
    # 2. פילטרים (Sobel Kernels)
    kernelY = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    
    kernelX = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    

