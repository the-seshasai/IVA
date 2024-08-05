import numpy as np
from PIL import Image
import cv2
def mse(imageA, imageB):
    # Ensure the images are numpy arrays
    imageA = np.array(imageA)
    imageB = np.array(imageB)
    
    # Check if the images have the same shape
    if imageA.shape != imageB.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # Compute the MSE
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    
    return err


# Load images using the provided helper functions or directly with cv2 or PIL
img1 = cv2.imread("./frames/I_frame_0381.jpeg") 
img2 = cv2.imread("./frames/B_frame_0002.jpeg") 

# Calculate MSE
mse_value = mse(img1, img2)
print(f"MSE between the images: {mse_value}")
