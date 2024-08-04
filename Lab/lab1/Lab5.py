import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt

# Load the image
image_path = 'img4.jpg'
image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for segmentation
color_ranges = {
    'orange': ([5, 50, 50], [15, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'green': ([40, 50, 50], [70, 255, 255]),
    'blue': ([100, 50, 50], [140, 255, 255]),
    'red1': ([0, 50, 50], [10, 255, 255]),
    'red2': ([170, 50, 50], [180, 255, 255])
}

# Segment the image based on color ranges
segmented_images = {}
for color, (lower, upper) in color_ranges.items():
    lower_bound = np.array(lower, dtype="uint8")
    upper_bound = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    segmented_images[color] = mask

# Combine the two red masks
red_mask = segmented_images['red1'] | segmented_images['red2']
segmented_images['red'] = red_mask
del segmented_images['red1']
del segmented_images['red2']

# Extract features from the segmented objects
features = []
for color, mask in segmented_images.items():
    # Perform noise removal and watershed algorithm
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    # Extract features
    labeled_image = label(markers > 1)
    regions = regionprops(labeled_image)
    for region in regions:
        features.append({
            'Color': color,
            'Area': region.area,
            'Perimeter': region.perimeter,
            'Eccentricity': region.eccentricity,
            'Solidity': region.solidity
        })

features_df = pd.DataFrame(features)

# Count the number of segmented objects
object_count = len(features_df)

# Display the results
print(features_df)

# Save DataFrame to a CSV file
features_df.to_csv('colored_segmented_features.csv', index=False)

# Display the original and segmented images
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(122)
plt.imshow(markers, cmap='jet')
plt.title('Segmented Image with Watershed')
plt.show()
