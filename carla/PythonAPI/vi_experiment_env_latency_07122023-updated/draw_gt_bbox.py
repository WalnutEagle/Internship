import numpy as np
import cv2

import os
from pathlib import Path

from pycocotools import mask
from skimage import measure

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

data_path = Path('raw_images/path_points_t10_32_95_7')
save_path = Path('raw_images/path_points_t10_32_95_7')

files = os.listdir(str(data_path))

num = int(len(files) / 3)
#num = int((len(files)-2) / 3)

for _nd in range(num):

	rgb = cv2.imread(str(data_path / ('%012d_front.png' % _nd)))


	

	
	width = 1024
	height = 576

	depth_threshold = 100

	fov = 120
	thresh = 0.2

	max_match_dis = 1
	max_match_dis_aids = 10
	max_match_dis_rider = 2

	focal = width / (2.0 * np.tan(fov * np.pi / 360.0))










	bounding_boxes, bboxes_3d, upper_bound, lower_bound, obj_ids, camera_matrix, in_vehicles, in_meters, _array_ins_rsp, _array_ins = np.load(str(data_path / ('%012d_veh.npy' % _nd)), allow_pickle=True).tolist()

	annotations = []
	cate_id = 10

	_index = np.zeros((height,width))

	_instances = []
	for a in _array_ins_rsp:
		if a not in _instances and a[0] == cate_id:
			_instances.append(a)

	m = len(upper_bound)
	assert m == len(obj_ids)
	# print('m: ', m)

	if m != 0:

		# bboxes_3d m,3,8
		bboxes_3d = np.array(bboxes_3d)
		# print('bboxes_3d: ', bboxes_3d.shape)
		assert bboxes_3d.shape[0] == m

		assert len(bounding_boxes) == len(obj_ids)

		pool = []

		for _insk, ins in enumerate(_instances):
			binary_mask = np.zeros((height,width))
			_index[(_array_ins[:,:,0]==ins[0])*(_array_ins[:,:,1]==ins[1])*(_array_ins[:,:,2]==ins[2])] = _insk+1
			binary_mask[(_array_ins[:,:,0]==ins[0])*(_array_ins[:,:,1]==ins[1])*(_array_ins[:,:,2]==ins[2])] = 1

			ins_h = np.where(binary_mask==1)[0]
			ins_w = np.where(binary_mask==1)[1]
			ins_d = in_meters[binary_mask==1]

			if np.min(ins_d) > depth_threshold:
				continue

			H = (((height/2.)-ins_h)*ins_d)/focal # z
			W = ((ins_w-(width/2.))*ins_d)/focal # y
			ins_3d = np.array([ins_d,W,H]).transpose() # nx3

			n = ins_3d.shape[0]

			ins3d_ext = np.tile(ins_3d,(m,1)) # (nxm,3)
			up_ext = np.tile(np.array(upper_bound),n).reshape((n*m,3))
			lo_ext = np.tile(np.array(lower_bound),n).reshape((n*m,3))


			ckup = ins3d_ext < up_ext
			cklo = ins3d_ext > lo_ext

			_ckup = np.prod(ckup,axis=1).reshape((m,n))
			_cklo = np.prod(cklo,axis=1).reshape((m,n))
			_ck = np.sum((_cklo * _ckup), axis=1) # (m,)

			_ind = np.where(_ck==max(_ck))[0]


			if max(_ck) == 0 or max(_ck)/n < thresh:
				ins_3d_tile = np.tile(ins_3d.reshape((n,3,1)),8)
				_ins_3d_tile = np.tile(ins_3d_tile,(1,m,1)).reshape((n*m,3,8))
				bboxes_3d_tile = np.tile(bboxes_3d,(n,1,1))

				dist = np.sqrt(np.sum((_ins_3d_tile-bboxes_3d_tile)**2, axis=1))

				_dist = dist.reshape((n,m,8))
				match = (_dist < max_match_dis)*1
				_match = (np.sum(match,axis=2)>=1)*1

				match_id = np.argmax(np.sum(_match,axis=0)/n)

				if (np.sum(_match,axis=0)/n)[match_id]>=thresh:
					ind = np.array([match_id])
				else:
					continue

			if_repeated = False
			if_multimatch = False

			if _ind.shape != (1,):
				print(_ind)
				print('Multiple matches!')
				if_multimatch = True
				_ind = np.array([_ind[0]])
				# continue

			assert _ind.shape == (1,)

			obj_id = obj_ids[_ind.item()]

			if _ind.item() in pool:
				if_repeated = True
				print('Repeated!')
				# assert False
			pool.append(_ind.item())

			minh = min(ins_h)
			maxh = max(ins_h)
			minw = min(ins_w)
			maxw = max(ins_w)
			bboxwidth = maxw - minw
			bboxheight = maxh - minh
			bbox = [minw, minh, bboxwidth, bboxheight]

			_segmentation = binary_mask_to_polygon(binary_mask, 1)
			binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
			area = mask.area(binary_mask_encoded)


			annotation_info = {'bbox': bbox,}

			annotations.append(annotation_info)



	for anno in annotations:
		bbox = anno['bbox']
		rgb = cv2.rectangle(rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,0,255), 2)



	bounding_boxes, bboxes_3d, upper_bound, lower_bound, obj_ids, camera_matrix, in_vehicles, in_meters, _array_ins_rsp, _array_ins = np.load(str(data_path / ('%012d_ped.npy' % _nd)), allow_pickle=True).tolist()
	

	annotations = []

	cate_id = 4

	_index = np.zeros((height,width))

	_instances = []
	for a in _array_ins_rsp:
		if a not in _instances and a[0] == cate_id:
			_instances.append(a)

	m = len(upper_bound)
	assert m == len(obj_ids)
	# print('m: ', m)

	if m != 0:

		# bboxes_3d m,3,8
		bboxes_3d = np.array(bboxes_3d)
		# print('bboxes_3d: ', bboxes_3d.shape)
		assert bboxes_3d.shape[0] == m

		assert len(bounding_boxes) == len(obj_ids)

		pool = []

		for _insk, ins in enumerate(_instances):
			binary_mask = np.zeros((height,width))
			_index[(_array_ins[:,:,0]==ins[0])*(_array_ins[:,:,1]==ins[1])*(_array_ins[:,:,2]==ins[2])] = _insk+1
			binary_mask[(_array_ins[:,:,0]==ins[0])*(_array_ins[:,:,1]==ins[1])*(_array_ins[:,:,2]==ins[2])] = 1

			ins_h = np.where(binary_mask==1)[0]
			ins_w = np.where(binary_mask==1)[1]
			ins_d = in_meters[binary_mask==1]

			if np.min(ins_d) > depth_threshold:
				continue

			H = (((height/2.)-ins_h)*ins_d)/focal # z
			W = ((ins_w-(width/2.))*ins_d)/focal # y
			ins_3d = np.array([ins_d,W,H]).transpose() # nx3

			n = ins_3d.shape[0]

			ins3d_ext = np.tile(ins_3d,(m,1)) # (nxm,3)
			up_ext = np.tile(np.array(upper_bound),n).reshape((n*m,3))
			lo_ext = np.tile(np.array(lower_bound),n).reshape((n*m,3))


			ckup = ins3d_ext < up_ext
			cklo = ins3d_ext > lo_ext

			_ckup = np.prod(ckup,axis=1).reshape((m,n))
			_cklo = np.prod(cklo,axis=1).reshape((m,n))
			_ck = np.sum((_cklo * _ckup), axis=1) # (m,)

			_ind = np.where(_ck==max(_ck))[0]


			if max(_ck) == 0 or max(_ck)/n < thresh:
				ins_3d_tile = np.tile(ins_3d.reshape((n,3,1)),8)
				_ins_3d_tile = np.tile(ins_3d_tile,(1,m,1)).reshape((n*m,3,8))
				bboxes_3d_tile = np.tile(bboxes_3d,(n,1,1))

				dist = np.sqrt(np.sum((_ins_3d_tile-bboxes_3d_tile)**2, axis=1))

				_dist = dist.reshape((n,m,8))
				match = (_dist < max_match_dis)*1
				_match = (np.sum(match,axis=2)>=1)*1

				match_id = np.argmax(np.sum(_match,axis=0)/n)

				if (np.sum(_match,axis=0)/n)[match_id]>=thresh:
					ind = np.array([match_id])
				else:
					continue

			if_repeated = False
			if_multimatch = False

			if _ind.shape != (1,):
				print(_ind)
				print('Multiple matches!')
				if_multimatch = True
				_ind = np.array([_ind[0]])
				# continue

			assert _ind.shape == (1,)

			obj_id = obj_ids[_ind.item()]

			if _ind.item() in pool:
				if_repeated = True
				print('Repeated!')
				# assert False
			pool.append(_ind.item())

			minh = min(ins_h)
			maxh = max(ins_h)
			minw = min(ins_w)
			maxw = max(ins_w)
			bboxwidth = maxw - minw
			bboxheight = maxh - minh
			bbox = [minw, minh, bboxwidth, bboxheight]

			_segmentation = binary_mask_to_polygon(binary_mask, 1)
			binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
			area = mask.area(binary_mask_encoded)


			annotation_info = {'bbox': bbox,}

			annotations.append(annotation_info)



	for anno in annotations:
		bbox = anno['bbox']
		rgb = cv2.rectangle(rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255,0,0), 2)



	cv2.imwrite(str(save_path / ('%012d_front_bbox.png' % _nd)), rgb)
