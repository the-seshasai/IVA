import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Paths to folders
frames_folder = 'frames'  
histogram_folder = 'histograms' 


os.makedirs(histogram_folder, exist_ok=True)


for frame_file in sorted(os.listdir(frames_folder)):
    if frame_file.endswith('.jpeg') or frame_file.endswith('.png'):

        frame_path = os.path.join(frames_folder, frame_file)
        color_image = cv2.imread(frame_path, cv2.IMREAD_COLOR)

        grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        histogram, bins = np.histogram(grayscale_image.flatten(), 256, [0,256])

        cdf = histogram.cumsum()

        cdf_normalized = cdf / cdf.max()

        threshold_value = np.argmax(cdf_normalized > 0.5)


        plt.figure(figsize=(8, 6))
        plt.hist(grayscale_image.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.7)
        plt.axvline(x=threshold_value, color='r', linestyle='--', linewidth=1.5)
        plt.title(f'Histogram for {frame_file}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        histogram_path = os.path.join(histogram_folder, f'histogram_{os.path.splitext(frame_file)[0]}.png')
        plt.savefig(histogram_path)
        plt.close()

        print(f'Histogram for {frame_file} saved to {histogram_path}')
