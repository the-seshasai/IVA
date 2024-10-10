import os
import random
import shutil

# Set paths
base_dir = r"C:\Users\SESHA\Downloads\drive-download-20240929T142842Z-001"
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')
train_images_dir = os.path.join(base_dir, 'train', 'images')
val_images_dir = os.path.join(base_dir, 'val', 'images')
train_labels_dir = os.path.join(base_dir, 'train', 'labels')
val_labels_dir = os.path.join(base_dir, 'val', 'labels')

# Create directories if not exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Set the split ratio
split_ratio = 0.8  # 80% for training and 20% for validation

# List all image files
all_images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
all_labels = [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))]

# Ensure corresponding images and labels match
all_images.sort()
all_labels.sort()

# Check if the number of images and labels are equal
assert len(all_images) == len(all_labels), "The number of images and labels do not match!"

# Shuffle and split
combined = list(zip(all_images, all_labels))
random.shuffle(combined)
train_size = int(len(combined) * split_ratio)
train_files = combined[:train_size]
val_files = combined[train_size:]

# Function to copy files to the respective directories
def copy_files(file_list, images_dest, labels_dest):
    for img_file, lbl_file in file_list:
        shutil.copy(os.path.join(images_dir, img_file), images_dest)
        shutil.copy(os.path.join(labels_dir, lbl_file), labels_dest)

# Copy train files
copy_files(train_files, train_images_dir, train_labels_dir)

# Copy val files
copy_files(val_files, val_images_dir, val_labels_dir)

print("Dataset split into train and val sets successfully!")
