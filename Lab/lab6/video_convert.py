import cv2
import os

# Function to create a video from frames
def create_video_from_frames(frame_folder, output_video_path, fps=30):
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])
    
    # Get the first frame to define the video size
    first_frame_path = os.path.join(frame_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec based on the required format
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Error reading frame {frame_file}")
            continue
        
        # Write each frame into the video
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video created and saved at {output_video_path}")

# Path to the folder containing marked frames
marked_frames_folder = 'marked_frames'
# Path to save the output video
output_video_path = 'output_video.avi'

# Create video from the marked frames
create_video_from_frames(marked_frames_folder, output_video_path, fps=30)
