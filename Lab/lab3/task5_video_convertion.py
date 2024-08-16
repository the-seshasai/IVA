import cv2
import os


output_dir = 'output'
video_output = 'output_video.avi'  


file_list = sorted([f for f in os.listdir(output_dir) if f.endswith(( '.jpeg'))])


if len(file_list) == 0:
    print("No images found in the output directory.")
    exit()


first_image_path = os.path.join(output_dir, file_list[0])
first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
height, width = first_image.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out = cv2.VideoWriter(video_output, fourcc, 20.0, (width, height), isColor=False)  


for filename in file_list:
    image_path = os.path.join(output_dir, filename)
    frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if frame is None:
        continue 
    
    out.write(frame) 

out.release()

print(f"Video saved as {video_output}")
