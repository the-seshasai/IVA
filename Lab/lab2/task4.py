import os

# Directory containing the renamed frames
frames_dir = './frames/'

# Initialize dictionaries to store sizes and counts
frame_sizes = {'I': [], 'P': [], 'B': []}

# Calculate the file sizes
for filename in os.listdir(frames_dir):
    if filename.startswith('I_') or filename.startswith('P_') or filename.startswith('B_'):
        frame_type = filename.split('_')[0]
        file_path = os.path.join(frames_dir, filename)
        file_size = os.path.getsize(file_path)
        frame_sizes[frame_type].append(file_size)

# Calculate average sizes
average_sizes = {frame_type: sum(sizes) / len(sizes) if sizes else 0 for frame_type, sizes in frame_sizes.items()}

# Print out the results
print("Average File Sizes (bytes):")
for frame_type, avg_size in average_sizes.items():
    print(f"{frame_type}: {avg_size:.2f} bytes")
