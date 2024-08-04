import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
gray_image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

# Display the image
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.show()


# Apply global thresholding
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Display the thresholded image
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholded Image')
plt.show()

