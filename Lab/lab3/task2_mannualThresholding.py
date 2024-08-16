import cv2
import os
import numpy as np
#In this method, we manually select a threshold value based on visual inspection. For instance, if the chosen threshold is 128
# Paths to folders
frames_folder = 'frames'  
binary_folder = 'ManualThershold'  

# Ensure the output folder exists
os.makedirs(binary_folder, exist_ok=True)

# Manual threshold value
threshold_value = 128  # You can adjust this value

# Process each frame
for frame_file in sorted(os.listdir(frames_folder)):
    if frame_file.endswith('.jpeg'):

        frame_path = os.path.join(frames_folder, frame_file)
        color_image = cv2.imread(frame_path, cv2.IMREAD_COLOR)

   
        grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)


        _, binary_image = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)


        binary_image_path = os.path.join(binary_folder, f'binary_{frame_file}')
        cv2.imwrite(binary_image_path, binary_image)

        print(f'Processed {frame_file}, binary image saved.')

