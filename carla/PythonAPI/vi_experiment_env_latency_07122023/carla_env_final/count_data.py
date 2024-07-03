import os

# Define the path to the "collection_data" folder
base_folder = "./"
count=0
# Loop through each subfolder in the "collection_data" folder
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)

    # Check if the item in the subfolder is a directory
    if os.path.isdir(subfolder_path):
        # Count the files in the subfolder
        file_count = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
        count+=file_count
        print(f"Subfolder '{subfolder}' contains {file_count} files.")

print(count)