import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os

# Step 1: Segment the Image (for creating reference features)
def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresholded

# Step 2: Identify the Objects of Interest and Extract Features
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

# Step 3: Create a Reference Feature Set for Oranges and Green Apples
def create_reference_features():
    reference_images = {
        "orange": "orange.jpg",
        "green_apple": "green.jpg"
    }
    reference_features = {}

    for label, img_path in reference_images.items():
        features_csv = f"{label}_features.csv"
        if os.path.exists(features_csv):
            # Load features if already saved
            print(f"Loading saved features for {label} from {features_csv}")
            reference_features[label] = pd.read_csv(features_csv, converters={"Color Histogram": eval})
        else:
            # Extract and save features
            print(f"Extracting and saving features for {label}")
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Image {img_path} not found.")
                continue
            thresholded = segment_image(image)
            features_df = extract_features(image, thresholded)
            features_df.to_csv(features_csv, index=False)
            reference_features[label] = features_df

    return reference_features

# Step 4: Extract Features from a Given Bounding Box
def extract_features_from_bbox(image, bbox):
    x, y, w, h = bbox
    cropped_image = image[y:y+h, x:x+w]
    mask = np.ones(cropped_image.shape[:2], dtype="uint8") * 255
    hist = cv2.calcHist([cropped_image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Step 5: Compare Features for Object Detection
def compare_features(reference_features, target_histogram):
    best_match = None
    best_distance = float('inf')

    for label, ref_features_df in reference_features.items():
        for ref_index, ref_row in ref_features_df.iterrows():
            ref_histogram = np.array(ref_row["Color Histogram"])
            dist = distance.euclidean(ref_histogram, target_histogram)

            if dist < best_distance:
                best_distance = dist
                best_match = label

    return best_match, best_distance


# Step 6: Detect and Label Object in a Provided Bounding Box
def detect_and_label_bbox(image, bbox, reference_features):
    target_histogram = extract_features_from_bbox(image, bbox)
    label, similarity_index = compare_features(reference_features, target_histogram)

    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label_text = f"{label} ({similarity_index:.2f})"
    cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image, similarity_index


# Step 7: Create Reference Features
reference_features = create_reference_features()

# Step 8: Load the Target Image
target_image_path = "images.jpeg"
target_image = cv2.imread(target_image_path)

# Let the user select the bounding box
if target_image is None:
    print(f"Error: Image {target_image_path} not found.")
else:
    bbox = cv2.selectROI("Select Bounding Box", target_image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Bounding Box")
    if bbox is not None:
        labeled_image, similarity_index = detect_and_label_bbox(target_image, bbox, reference_features)

        # Save and display the result
        detected_image_path = "detected_and_labeled_target.jpg"
        cv2.imwrite(detected_image_path, labeled_image)

        # Display the result
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected and Labeled Object - Similarity Index: {similarity_index:.2f}")
        plt.axis('off')
        plt.show()

        print(f"Detected and labeled object saved as {detected_image_path}")
        print(f"Similarity index: {similarity_index:.2f}")

