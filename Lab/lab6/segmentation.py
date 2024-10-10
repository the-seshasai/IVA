import cv2
import os
import numpy as np

# Function to display HSV values for manual tuning
def display_hsv_values(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('Original Frame', frame)
    cv2.imshow('HSV Frame', hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to perform color threshold segmentation
def color_threshold_segmentation(frame, lower_color, upper_color):
    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply threshold
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Return the segmented mask
    return mask

# Path to the folder containing extracted frames
extracted_frames_folder = 'extracted_frames'
# Path to the folder to save segmented frames
segmented_frames_folder = 'segmented_frames'

# Create the segmented frames folder if it doesn't exist
if not os.path.exists(segmented_frames_folder):
    os.makedirs(segmented_frames_folder)

# Get the list of frame files in the extracted frames folder
frame_files = sorted([f for f in os.listdir(extracted_frames_folder) if f.endswith('.png')])

# Define an initial HSV range for segmentation (you can adjust these)
lower_color = np.array([0, 50, 50])  # Example: Adjust these values
upper_color = np.array([180, 255, 255])

# Perform segmentation on each frame and save the result
for frame_file in frame_files:
    # Read the frame
    frame_path = os.path.join(extracted_frames_folder, frame_file)
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"Error reading frame {frame_file}")
        continue

    # Display the HSV values to manually adjust
    display_hsv_values(frame)
    
    # Apply segmentation with the manually adjusted color range
    segmented_frame = color_threshold_segmentation(frame, lower_color, upper_color)
    
    # Save the segmented frame
    segmented_frame_path = os.path.join(segmented_frames_folder, frame_file)
    cv2.imwrite(segmented_frame_path, segmented_frame)

print(f"Segmented frames saved in '{segmented_frames_folder}'")
