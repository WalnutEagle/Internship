import os
import glob
import numpy as np

# Define the path to the "collection_data" folder
base_folder = "./collection_data_13_11/"
count=0
img_list=[]
data_list=[]
img_list_final=[]
data_list_final=[]
# Loop through each subfolder in the "collection_data" folder
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)

    # Check if the item in the subfolder is a directory
    if os.path.isdir(subfolder_path):
        img_list =  img_list+glob.glob(subfolder_path+'/*.jpg')
        data_list = data_list+glob.glob(subfolder_path+'/*.npy')
        # Count the files in the subfolder
        # file_count = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
        # count+=file_count
        #print(f"Subfolder '{subfolder}' contains {file_count} files.")
img_list=sorted(img_list)
data_list=sorted(data_list)
for i in data_list:
    val=np.load(i)
    if val[1]==0.0:
        data_list_final.append(i)
        img_list_final.append(i.split('.')[-2]+'.jpg')



# print(img_list[3])
# print(data_list[3])