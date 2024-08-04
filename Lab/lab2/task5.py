import cv2
import os

# Path to the directory containing I frames
i_frame_dir = './I_frames/'  # Update this path as needed
output_video_path = 'reconstructed.mp4'  # Output path for the reconstructed video

# Define frame rate (we'll use 3.5 fps for at least 2 seconds duration)
frame_rate = 3.5
num_frames = 7

# Get the list of frame file names
frame_files = [f for f in sorted(os.listdir(i_frame_dir)) if f.startswith('I_frame_')]

# Check if the number of frames matches the expected count
if len(frame_files) != num_frames:
    print(f"Error: Expected {num_frames} frames, found {len(frame_files)}.")
else:
    # Load the first frame to get the frame size
    first_frame = cv2.imread(os.path.join(i_frame_dir, frame_files[0]))
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(i_frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video created successfully and saved as {output_video_path}")
