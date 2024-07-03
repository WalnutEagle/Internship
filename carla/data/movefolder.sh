#!/bin/bash

# Source directory where .npy files are located
src_directory="/data2/kathakoli/carla/data/ECCV_img12/session_5/Town10HD/004/wk4"

# Destination directory where you want to move the .npy files
dest_directory="/data2/kathakoli/carla/imitation_data_final"

# Extract the base name of the destination directory

src_dir_name=$(echo $src_directory | tr '/' '_')

# Ensure the destination directory exists, create it if it doesn't
mkdir -p "$dest_directory"

# Use the find command to locate .npy files in the source directory and move them to the destination with the new name format
find "$src_directory" -type f -name "*_front.npy" -exec bash -c 'cp "$1" "$2/$(basename "$1" .npy)_'"$src_dir_name"'.npy"' _ {} "$dest_directory" \;
find "$src_directory" -type f -name "*_front.jpg" -exec bash -c 'cp "$1" "$2/$(basename "$1" .jpg)_'"$src_dir_name"'.jpg"' _ {} "$dest_directory" \;

echo "All .npy files moved to $dest_directory with the new names"