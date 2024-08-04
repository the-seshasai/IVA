import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('img2.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.imshow(image_rgb)
plt.title('Original Image')
plt.show()

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for segmentation
color_ranges = {
    'red': [(0, 120, 70), (10, 255, 255)],
    'green': [(36, 100, 100), (86, 255, 255)],
    'blue': [(94, 80, 2), (126, 255, 255)],
    'yellow': [(15, 100, 100), (35, 255, 255)],
    'cyan': [(78, 100, 100), (88, 255, 255)]
}

# Initialize a dictionary to store segmented images
segmented_images = {}

# Apply color thresholding for each color
for color, (lower_bound, upper_bound) in color_ranges.items():
    mask = cv2.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))
    segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    segmented_images[color] = segmented_image

    # Display the segmented image for each color
    plt.imshow(segmented_image)
    plt.title(f'Segmented Image - {color}')
    plt.show()

# Display original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

for i, (color, segmented_image) in enumerate(segmented_images.items(), start=2):
    plt.subplot(2, 3, i)
    plt.imshow(segmented_image)
    plt.title(f'Segmented - {color}')

plt.tight_layout()
plt.show()
