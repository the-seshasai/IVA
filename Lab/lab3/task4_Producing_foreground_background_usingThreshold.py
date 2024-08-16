import cv2
import os
import matplotlib.pyplot as plt


input_dir = 'frames'
output_dir = 'output'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for filename in os.listdir(input_dir):
    if filename.endswith(('.jpeg')):  

        image_path = os.path.join(input_dir, filename)
        color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if color_image is None:
            continue  
        

        grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)


        _, otsu_binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_threshold = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, otsu_binary_image)

