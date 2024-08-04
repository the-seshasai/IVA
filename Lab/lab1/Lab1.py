#Read the Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('img.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image_rgb)
plt.title('Original Image')
plt.show()

#Compute Basic Statistics
# Compute mean and standard deviation for each channel
means = cv2.mean(image)
std_devs = cv2.meanStdDev(image)[1]

print(f'Means: {means}')
print(f'Standard Deviations: {std_devs}')

# Compute and display histograms for each channel
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.title('Histograms for B, G, R channels')
plt.show()
#Convert Color Spaces
# Convert to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_RGB2HSV)
plt.imshow(hsv_image_rgb)
plt.title('HSV Image')
plt.show()

# Convert to Lab
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
lab_image_rgb = cv2.cvtColor(lab_image, cv2.COLOR_RGB2LAB)
plt.imshow(lab_image_rgb)
plt.title('Lab Image')
plt.show()

