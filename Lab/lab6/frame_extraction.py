import cv2
import os

# Load video
video_path = 'in.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")

# Create a directory to save frames
output_folder = 'extracted_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Extract frames and save them
frame_list = []
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Save each frame as an image file in the folder
        frame_filename = f"{output_folder}/frame_{frame_count:04d}.png"
        cv2.imwrite(frame_filename, frame)
        frame_list.append(frame)
        frame_count += 1
    else:
        break

cap.release()

print(f"Saved {frame_count} frames to the folder '{output_folder}'")
