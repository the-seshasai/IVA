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
def detect_soft_cuts(frames, threshold=0.998):
    cuts = []
    for i in range(1, len(frames)):
        hist1 = cv2.calcHist([frames[i-1]], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frames[i]], [0], None, [256], [0, 256])
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        if hist_diff < threshold:
            cuts.append(i)
    return cuts

# Function to detect an object in a frame using contours
def detect_object(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)  # Bounding box of the object
    else:
        return None

# Function to track an object between frames, re-initialize after cuts
def track_object(frames, hard_cuts, soft_cuts):
    tracker = cv2.TrackerCSRT_create()  # You can change the tracker to KCF or MOSSE if needed
    tracked_frames = []
    init_bbox = None
    reinitialize_tracker = True
    
    for i in range(len(frames)):
        # Reinitialize the tracker after a cut
        if i in hard_cuts or i in soft_cuts or reinitialize_tracker:
            bbox = detect_object(frames[i])
            if bbox:
                tracker = cv2.TrackerCSRT_create()  # Reinitialize the tracker
                tracker.init(frames[i], bbox)
                init_bbox = bbox
                reinitialize_tracker = False
            else:
                print(f"No object detected in frame {i}.")
                reinitialize_tracker = True
                tracked_frames.append(frames[i])
                continue
        
        # Track the object
        success, bbox = tracker.update(frames[i])
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            frames[i] = cv2.rectangle(frames[i], p1, p2, (255, 0, 0), 2, 1)
            cv2.putText(frames[i], "Tracking", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        else:
            cv2.putText(frames[i], "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            reinitialize_tracker = True  # If tracking fails, reinitialize on the next frame
        tracked_frames.append(frames[i])
    return tracked_frames

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
marked_frames_folder = 'marked_frames_track'

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

# Detect hard and soft cuts
hard_cuts = detect_hard_cuts(frames)
soft_cuts = detect_soft_cuts(frames)

# Mark the frames where cuts are detected
marked_frames = mark_cuts(frames, hard_cuts, soft_cuts)

# Track the object across frames, reinitializing after cuts
tracked_frames = track_object(marked_frames, hard_cuts, soft_cuts)

# Save the tracked frames with marked cuts
for i, frame in enumerate(tracked_frames):
    marked_frame_path = os.path.join(marked_frames_folder, frame_files[i])
    cv2.imwrite(marked_frame_path, frame)

print(f"Hard cuts detected at: {hard_cuts}")
print(f"Soft cuts detected at: {soft_cuts}")
print(f"Tracked and marked frames saved in '{marked_frames_folder}'")
