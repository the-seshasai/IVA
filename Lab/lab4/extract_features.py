# extract_features.py

import cv2
import numpy as np
import pandas as pd

def extract_features(image, thresholded):
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = {
        "Area": [],
        "Perimeter": [],
        "Bounding Box": [],
        "Centroid": [],
        "Color Histogram": []
    }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for contour in contours:
        # Shape Features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Color Features
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        hist = cv2.calcHist([masked_img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Store features
        features["Area"].append(area)
        features["Perimeter"].append(perimeter)
        features["Bounding Box"].append((x, y, w, h))
        features["Centroid"].append((cX, cY))
        features["Color Histogram"].append(hist.tolist())

    return pd.DataFrame(features)

if __name__ == "__main__":
    image_path = "orange.jpg"
    thresholded_path = "thresholded_image.jpg"
    
    image = cv2.imread(image_path)
    thresholded = cv2.imread(thresholded_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or thresholded is None:
        print("Error: Required image or thresholded image not found.")
    else:
        features_df = extract_features(image, thresholded)
        features_df.to_csv("orange_features.csv", index=False)
        print("Features saved as orange_features.csv")
