from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def load_image(path):
    # טעינת התמונה
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
    
    # 3. קונבולוציה
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    
    # 4. חישוב עוצמה (Magnitude)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG

# --- הרצת התהליך ---

# א. טעינת התמונה (וודא שהקובץ נמצא בתיקייה שלך בקולאב)
img_maya = load_image('image1.jpeg')

# ב. הפעלת זיהוי הקצוות
show_image = edge_detection(img_maya)

# ג. הצגת התוצאה
plt.figure(figsize=(10, 10))
plt.imshow(show_image, cmap='gray')
plt.title("Edge Detection Result (Sobel)")
plt.axis('off')
plt.show()
