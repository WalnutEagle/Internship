import glob
import numpy as np

src_dest = "/data2/kathakoli/carla/data/ECCV_img12/session_5/Town10HD/004/"
for walk in ['wk1','wk2','wk3','wk4']:
    #print(src_dest)#,
    files = glob.glob(src_dest+ walk + '/*_front.npy')
    prev_angle=0.0
    print(files)
    sorted_files=sorted(files)
    for i in range(len(sorted_files)):
        file_name=sorted_files[i].split('.')[-2]
        array=np.load(file_name+'.npy')
        array[4]=array[4]+prev_angle
        prev_angle=array[4]
        np.save(file_name+'.npy',array)
        print(file_name)
    



