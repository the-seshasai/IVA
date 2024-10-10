import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance
from create_reference_features import create_reference_features

# Extract features from a circular region (instead of a bounding box)
def extract_features_from_circle(image, center, radius):
    # Create a mask for the circular region
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
    hist = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Compare features between the target and reference images
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

if __name__ == "__main__":
    # Load the reference features for comparison
    reference_features = create_reference_features()

    # Load the target image
    target_image_path = "images.jpeg"
    image = cv2.imread(target_image_path)

    if image is None:
        print(f"Error: Image {target_image_path} not found.")
    else:
        rois = []  # List to store selected ROIs
        while True:
            # Select multiple regions of interest (ROI) using OpenCV's ROI selector
            bbox = cv2.selectROI("Select Object", image, fromCenter=False, showCrosshair=True)
            
            # Check if a valid ROI has been selected
            if bbox == (0, 0, 0, 0):
                break  # Exit if ESC or an invalid region is selected

            # Add the selected bounding box to the list
            rois.append(bbox)

        # Process each selected ROI
        for i, bbox in enumerate(rois):
            # Extract the selected region of interest (ROI)
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find contours within the thresholded region
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour and calculate the enclosing circle
                largest_contour = max(contours, key=cv2.contourArea)
                (x_center, y_center), radius = cv2.minEnclosingCircle(largest_contour)
                
                # Adjust the center to the original image's coordinates
                center = (int(x_center + x), int(y_center + y))
                radius = int(radius)

                # Extract features from the circular region
                target_histogram = extract_features_from_circle(image, center, radius)
                
                # Compare the extracted features with reference features
                label, similarity_index = compare_features(reference_features, target_histogram)
                
                # Draw the circle and its centroid on the image with adjusted colors and thickness
                cv2.circle(image, center, radius, (0, 255, 0), 3)  # Thicker circle
                cv2.circle(image, center, 5, (0, 0, 255), -1)  # Centroid in red
                
                # Adjust label placement: Move text above the object, or to the right if it's near the top
                text_position = (center[0] - radius, max(center[1] - radius - 20, 20))  # Avoid text going off the image
                label_text = f"{label} ({similarity_index:.2f})"
                
                # Adjust text formatting: smaller font size and thickness to ensure it's readable
                font_scale = 0.7
                font_thickness = 2

                # Draw the label text
                cv2.putText(image, label_text, text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

        # Save and display the result
        detected_image_path = "detected_and_labeled_target_circle_multiple.jpg"
        cv2.imwrite(detected_image_path, image)

        # Display the result using matplotlib
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Detected and Labeled Objects with Multiple Bounding Circles")
        plt.axis('off')
        plt.show()

        print(f"Detected and labeled image saved as {detected_image_path}")
