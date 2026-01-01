from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path_to_maya):
    path_to_maya = 'image1.jpeg'
    image_maya = Image.open(path_to_maya)
    image_maya = np.array(image_maya)

    plt.imshow(image_maya) # imshow plot images

def edge_detection(image_array):
    gray_image = np.mean(image_array, axis=2)
    
    # 2. פילטר לזיהוי שינויים אנכיים (עבור קצוות אנכיים)
    kernelY = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    
    # 3. פילטר לזיהוי שינויים אופקיים (עבור קצוות אופקיים)
    kernelX = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    # 4. ביצוע קונבולוציה עם Zero Padding (boundary='fill')
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    
    # 5. שילוב התוצאות לעוצמה כוללת (Magnitude)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG

# בדיקת הפונקציה והצגת התמונה
# (בהנחה ש-image_maya הוא מערך ה-numpy של התמונה שלך)
show_image = edge_detection(image_maya)

plt.figure(figsize=(8, 8))
plt.imshow(show_image, cmap='gray')
plt.title("Edge Detection Result (Sobel)")
plt.axis('off') # להסרת הצירים מהתצוגה
plt.show()
