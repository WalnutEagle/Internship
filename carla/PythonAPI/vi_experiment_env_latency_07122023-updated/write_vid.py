import os
import cv2
from pathlib import Path
import numpy as np

path = Path('/data2/jimuyang/redhat/vi_experiment_env_04032023/path_points_t10_32_95_8')
out = cv2.VideoWriter('/data2/jimuyang/redhat/vi_experiment_env_04032023/path_points_t10_32_95_8/demo_8.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 6, (1024,576))

num = int(len(os.listdir(str(path)))/3)

for ct in range(num):
	# if ct<62:
	# 	continue
	print(ct)
	front = cv2.imread(str(path / ('%012d_front.png' % ct)))
	# front = cv2.resize(front, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	# bev = cv2.imread(str(path / ('%012d_bev.png' % ct)))
	
	# veh = cv2.imread(str(path / ('%012d_veh.png' % ct)))
	# veh = cv2.resize(veh, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	# veh_seg = cv2.imread(str(path / ('%012d_vehseg.png' % ct)))
	# veh_seg = cv2.resize(veh_seg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

	# vis1 = np.hstack((bev, front))
	# vis2 = np.hstack((veh, veh_seg))
	# vis = np.vstack((vis1,vis2))
	# vis = np.uint8(vis)
	out.write(front)