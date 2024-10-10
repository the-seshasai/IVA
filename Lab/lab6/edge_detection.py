import cv2
import os

# Function to perform edge detection
def edge_detection(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray_frame, 100, 200)
    return edges

# Path to the folder containing segmented frames
segmented_frames_folder = 'segmented_frames'
# Path to the folder to save edge-detected frames
edge_frames_folder = 'edge_detected_frames'

# Create the folder for saving edge-detected frames if it doesn't exist
if not os.path.exists(edge_frames_folder):
    os.makedirs(edge_frames_folder)

# Get the list of frame files in the segmented frames folder
frame_files = sorted([f for f in os.listdir(segmented_frames_folder) if f.endswith('.png')])

# Perform edge detection on each frame and save the result
for frame_file in frame_files:
    # Read the segmented frame
    frame_path = os.path.join(segmented_frames_folder, frame_file)
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"Error reading frame {frame_file}")
        continue
    
    # Apply edge detection
    edges = edge_detection(frame)
    
    # Save the edge-detected frame
    edge_frame_path = os.path.join(edge_frames_folder, frame_file)
    cv2.imwrite(edge_frame_path, edges)

print(f"Edge-detected frames saved in '{edge_frames_folder}'")
