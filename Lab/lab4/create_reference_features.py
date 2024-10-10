# create_reference_features.py

import cv2
import pandas as pd
import os
from segment_image import segment_image
from extract_features import extract_features

def create_reference_features():
    reference_images = {
        "orange": "orange.jpg",
        "green_apple": "green.jpg"
    }
    reference_features = {}

    for label, img_path in reference_images.items():
        features_csv = f"{label}_features.csv"
        if os.path.exists(features_csv):
            print(f"Loading saved features for {label} from {features_csv}")
            reference_features[label] = pd.read_csv(features_csv, converters={"Color Histogram": eval})
        else:
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

if __name__ == "__main__":
    reference_features = create_reference_features()
    print("Reference features created and saved.")
