import cv2
import os
import numpy as np

# Function to calculate absolute pixel difference for hard cuts
def detect_hard_cuts(frames, threshold=500000):
    cuts = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        non_zero_count = np.count_nonzero(diff)
        if non_zero_count > threshold:
            cuts.append(i)
    return cuts

# Function to calculate histogram difference for soft cuts
# Hard cut frames are excluded from soft cut detection
def detect_soft_cuts(frames, hard_cuts, threshold=0.998):
    cuts = []
    for i in range(1, len(frames)):
        # Skip if the frame is a hard cut
        if i in hard_cuts:
            continue
        hist1 = cv2.calcHist([frames[i-1]], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frames[i]], [0], None, [256], [0, 256])
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        if hist_diff < threshold:
            cuts.append(i)
    return cuts

# Function to mark detected cuts in the frames
def mark_cuts(frames, hard_cuts, soft_cuts):
    marked_frames = []
    for i, frame in enumerate(frames):
        if i in hard_cuts:
            cv2.putText(frame, "Hard Cut", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
        elif i in soft_cuts:
            cv2.putText(frame, "Soft Cut", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
        marked_frames.append(frame)
    return marked_frames

# Path to the folder containing segmented frames
segmented_frames_folder = 'segmented_frames'
# Path to the folder to save marked frames with scene cuts
marked_frames_folder = 'marked_frames'

# Create folder to save the marked frames if it doesn't exist
if not os.path.exists(marked_frames_folder):
    os.makedirs(marked_frames_folder)

# Get the list of frame files in the segmented frames folder
frame_files = sorted([f for f in os.listdir(segmented_frames_folder) if f.endswith('.png')])

# Load all the frames into a list
frames = []
for frame_file in frame_files:
    frame_path = os.path.join(segmented_frames_folder, frame_file)
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error reading frame {frame_file}")
        continue
    frames.append(frame)

# Detect hard cuts
hard_cuts = detect_hard_cuts(frames)

# Detect soft cuts, excluding frames with hard cuts
soft_cuts = detect_soft_cuts(frames, hard_cuts)

# Mark the frames where cuts are detected
marked_frames = mark_cuts(frames, hard_cuts, soft_cuts)

# Save the marked frames
for i, frame in enumerate(marked_frames):
    marked_frame_path = os.path.join(marked_frames_folder, frame_files[i])
    cv2.imwrite(marked_frame_path, frame)

print(f"Hard cuts detected at: {hard_cuts}")
print(f"Soft cuts detected at: {soft_cuts}")
print(f"Marked frames saved in '{marked_frames_folder}'")
