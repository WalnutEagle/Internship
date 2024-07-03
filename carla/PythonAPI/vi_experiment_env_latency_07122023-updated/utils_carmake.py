from pathlib import Path
import carla
import random
import weakref
import math

import sys
import os

from pathlib import Path

import numpy as np
import tqdm
import carla
import cv2
import pandas as pd

from PIL import Image
import torch

import random
import pygame
from carla import TrafficLightState as tls

try:
    import queue
except ImportError:
    import Queue as queue

import time

from pycocotools import mask
from skimage import measure

COLOR_BANK = {1: [70, 70, 70], 2: [190, 153, 153], 3: [72, 0, 90], 4: [220, 20, 60], 5: [153, 153, 153],
        6: [157, 234, 50], 7: [128, 64, 128], 8: [244, 35, 232], 9: [107, 142, 35], 10: [0, 0, 255],
        11: [102, 102, 156], 12: [220, 220, 0], 13: [20, 20, 20], 14: [90, 90, 90], 15: [210, 173, 173],
        16: [92, 20, 110], 17: [240, 40, 80], 18: [173, 173, 173], 19: [177, 214, 70], 20: [148, 84, 148],
        21: [224, 55, 212], 22: [127, 162, 55],23: [20, 20, 235], 24: [122, 122, 176], 25: [240, 240, 20],
        26: [40, 40, 40], 27: [110, 110, 110], 28: [230, 193, 193], 29: [112, 40, 130], 30: [180, 60, 100],
        31: [193, 193, 193], 32: [197, 194, 90], 33: [168, 104, 168], 34: [204, 75, 192], 35: [147, 182, 75],
        36: [40, 40, 215], 37: [142, 142, 196], 38: [180, 180, 40], 39: [60, 60, 60], 40: [130, 130, 130],
        41: [250, 213, 213], 42: [132, 60, 150], 43: [160, 80, 120], 44: [213, 213, 213], 45: [217, 174, 110],
        46: [188, 144, 188], 47: [184, 95, 172], 48: [167, 202, 95], 49: [60, 60, 195], 50: [162, 162, 216],
        51: [160, 160, 60], 52: [80, 80, 80], 53: [150, 150, 150], 54: [110, 73, 73], 55: [152, 80, 170],
        56: [140, 100, 140], 57: [233, 233, 233], 58: [237, 154, 130], 59: [208, 144, 208], 60: [164, 115, 152],
        61: [187, 222, 115], 62: [80, 80, 175], 63: [182, 182, 226], 64: [140, 140, 80], 65: [100, 100, 100],
        66: [150, 150, 170], 67: [0, 73, 73], 68: [152, 80, 0], 69: [0, 100, 140], 70: [233, 0, 233]}

def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """

    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix



def draw_bounding_boxes_cv2(image, bounding_boxes, color):
    """
    Draws bounding boxes on image.
    """

    for k, bbox in enumerate(bounding_boxes):
        points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]


        # image = cv2.putText(image, str(k), points[0], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        

        cv2.line(image, points[0], points[1], color, 1)
        cv2.line(image, points[0], points[1], color, 1)
        cv2.line(image, points[1], points[2], color, 1)
        cv2.line(image, points[2], points[3], color, 1)
        cv2.line(image, points[3], points[0], color, 1)
        # top
        cv2.line(image, points[4], points[5], color, 1)
        cv2.line(image, points[5], points[6], color, 1)
        cv2.line(image, points[6], points[7], color, 1)
        cv2.line(image, points[7], points[4], color, 1)
        # base-top
        cv2.line(image, points[0], points[4], color, 1)
        cv2.line(image, points[1], points[5], color, 1)
        cv2.line(image, points[2], points[6], color, 1)
        cv2.line(image, points[3], points[7], color, 1)

    return image



# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        bounding_boxes = []
        in_vehicles = {}
        upper_bound = []
        lower_bound = []
        obj_ids = []
        bboxes_3d = []

        for _id, vehicle in enumerate(vehicles):

            _obj_id = vehicle.id

            camera_bbox, cords_x_y_z = ClientSideBoundingBoxes.get_bounding_box(vehicle, camera)\


            if any(camera_bbox[:, 2] <= 0):
                continue

            bounding_boxes.append(camera_bbox)
            # in_vehicles[_id+1] = (cords_x_y_z, camera_bbox, _obj_id) # index starts from 1
            in_vehicles[_obj_id] = (cords_x_y_z, camera_bbox)

            # print('cords_x_y_z: ', cords_x_y_z.shape)
            _cords_x_y_z = np.array(cords_x_y_z)

            bboxes_3d.append(_cords_x_y_z)

            _maxx = max(_cords_x_y_z[0,:])
            _minx = min(_cords_x_y_z[0,:])
            _maxy = max(_cords_x_y_z[1,:])
            _miny = min(_cords_x_y_z[1,:])
            _maxz = max(_cords_x_y_z[2,:])
            _minz = min(_cords_x_y_z[2,:])

            upper_bound.append([_maxx, _maxy, _maxz])
            lower_bound.append([_minx, _miny, _minz])
            obj_ids.append(_obj_id)

        # # filter objects behind camera
        # bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        # return bounding_boxes, in_vehicles, upper_bound, lower_bound, obj_ids
        camera_matrix = ClientSideBoundingBoxes.get_matrix(camera.get_transform())

        return bounding_boxes, bboxes_3d, upper_bound, lower_bound, obj_ids, camera_matrix, in_vehicles


    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox, cords_x_y_z

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        # print(world_cord.shape) # (4,8)
        # print(world_cord)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

def from_file(poses_txt):
    #pairs_file = Path(__file__).parent / poses_txt
    pairs_file = Path(poses_txt)
    pairs = pairs_file.read_text().strip().split('\n')
    pairs = [(int(x[0]), int(x[1])) for x in map(lambda y: y.split(), pairs)]

    return pairs

depth_threshold = 100
# width = 1280
# height = 720

width = 1024
height = 576


max_match_dis = 1
max_match_dis_aids = 10
max_match_dis_rider = 2

thresh = 0.2





def get_half_anno(actors, camera, ins_img, dep_img, annotations, cate_id, fov):

    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))

    actors_dict = {}
    for _a in actors:
        actors_dict[_a.id] = _a

    _array_dep = np.frombuffer(dep_img.raw_data, dtype=np.dtype("uint8"))
    _array_dep = np.reshape(_array_dep, (dep_img.height, dep_img.width, 4))
    _array_dep = _array_dep[:, :, :3]
    _array_dep = _array_dep[:, :, ::-1]
    depth = np.float32(_array_dep)
    normalized = (depth[:,:,0] + depth[:,:,1]*256 + depth[:,:,2]*256*256) / (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized

    bounding_boxes, bboxes_3d, upper_bound, lower_bound, obj_ids, camera_matrix, in_vehicles = ClientSideBoundingBoxes.get_bounding_boxes(actors,camera)

    _array_ins = np.frombuffer(ins_img.raw_data, dtype=np.dtype("uint8"))
    _array_ins = np.reshape(_array_ins, (ins_img.height, ins_img.width, 4))
    _array_ins = _array_ins[:, :, :3]
    _array_ins = _array_ins[:, :, ::-1]

    _index = np.zeros((height,width)) # instance index map but the index is not the object id from carla
    # print('shape: ', _index.shape, _array_ins.shape)
    _array_ins_rsp = _array_ins.reshape(-1,3).tolist()
    # _instances = []
    # for a in _array_ins_rsp:
    #     if a not in _instances and a[0] == cate_id:
    #         _instances.append(a)


    # print('_instances: ', len(_instances))

    # info = [bounding_boxes, bboxes_3d, upper_bound, lower_bound, obj_ids, camera_matrix, in_vehicles, in_meters, _instances]
    info = [bounding_boxes, bboxes_3d, upper_bound, lower_bound, obj_ids, camera_matrix, in_vehicles, in_meters, _array_ins_rsp, _array_ins]


    return info








def get_anno(actors, camera, ins_img, dep_img, annotations, cate_id, fov):

    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))

    # calibration = np.identity(3)
    # calibration[0, 2] = width / 2.0
    # calibration[1, 2] = height / 2.0
    # calibration[0, 0] = calibration[1, 1] = width / (2.0 * np.tan(fov * np.pi / 360.0))
    # camera.calibration = calibration
    
    actors_dict = {}
    for _a in actors:
        actors_dict[_a.id] = _a

    _array_dep = np.frombuffer(dep_img.raw_data, dtype=np.dtype("uint8"))
    _array_dep = np.reshape(_array_dep, (dep_img.height, dep_img.width, 4))
    _array_dep = _array_dep[:, :, :3]
    _array_dep = _array_dep[:, :, ::-1]
    depth = np.float32(_array_dep)
    normalized = (depth[:,:,0] + depth[:,:,1]*256 + depth[:,:,2]*256*256) / (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized

    bounding_boxes, bboxes_3d, upper_bound, lower_bound, obj_ids, camera_matrix, in_vehicles = ClientSideBoundingBoxes.get_bounding_boxes(actors,camera)

    _array_ins = np.frombuffer(ins_img.raw_data, dtype=np.dtype("uint8"))
    _array_ins = np.reshape(_array_ins, (ins_img.height, ins_img.width, 4))
    _array_ins = _array_ins[:, :, :3]
    _array_ins = _array_ins[:, :, ::-1]

    _index = np.zeros((height,width)) # instance index map but the index is not the object id from carla
    # print('shape: ', _index.shape, _array_ins.shape)
    _array_ins_rsp = _array_ins.reshape(-1,3).tolist()
    _instances = []
    for a in _array_ins_rsp:
        if a not in _instances and a[0] == cate_id:
            _instances.append(a)


    # print('_instances: ', len(_instances))


    m = len(upper_bound)
    assert m == len(obj_ids)
    # print('m: ', m)

    if m == 0:
        return annotations

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
                _ind = np.array([match_id])
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

        # print(_ind.item())

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


        # print(actors_dict[obj_id].attributes)

        attri = actors_dict[obj_id].attributes
        if 'color' in list(attri.keys()):
            _color = actors_dict[obj_id].attributes['color']
        else:
            _color = None

        annotation_info = {
                        'obj_id': obj_id,
                        'bbox': bbox,
                        'segmentation': _segmentation,
                        'area': area,
                        'if_repeated': if_repeated,
                        'if_multimatch': if_multimatch,
                        'camera_matrix': camera_matrix,
                        '3d': in_vehicles[obj_id],
                        'brand': actors_dict[obj_id].type_id,
                        'color': _color
                        }

        annotations.append(annotation_info)

    return annotations







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


two_veh_percent = 0.5
def spawn_vehicles(client, world, number_of_vehicles, synchronous_master, traffic_manager, car_lights_on=False):
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    vehicles_list = []
    spawn_points = world.get_map().get_spawn_points()
    batch = []
    vehicles_bp = world.get_blueprint_library().filter('vehicle.*')
    # vehicles_bp_4 = [x for x in vehicles_bp if int(x.get_attribute('number_of_wheels')) == 4] # only use 4 wheel vehicles
    # vehicles_bp_2 = [x for x in vehicles_bp if int(x.get_attribute('number_of_wheels')) == 2] # only use 2 wheel vehicles

    vehicles_bp_4 = []
    for x in vehicles_bp:
        if int(x.get_attribute('number_of_wheels')) != 4:
            continue
        if x.id in ['vehicle.micro.microlino', 'vehicle.dodge.charger_police', 'vehicle.ford.ambulance', 'vehicle.carlamotors.carlacola', 'vehicle.dodge.charger_police_2020', 'vehicle.carlamotors.firetruck']:
            continue
        vehicles_bp_4.append(x)

    # hero = args.hero
    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        # if random.random() > two_veh_percent:
        #     blueprint = random.choice(vehicles_bp_4)
        # else:
        #     blueprint = random.choice(vehicles_bp_2)

        blueprint = random.choice(vehicles_bp_4)

        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            print(response.error)
        else:
            vehicles_list.append(response.actor_id)

    # Set automatic vehicle lights update if specified
    if car_lights_on:
        all_vehicle_actors = world.get_actors(vehicles_list)
        for actor in all_vehicle_actors:
            traffic_manager.update_vehicle_lights(actor, True)

    return vehicles_list


def spawn_walkers(client, world, number_of_walkers, seedw, percentagePedestriansRunning=0.1, percentagePedestriansCrossing=0.1):
    SpawnActor = carla.command.SpawnActor

    walkers_list = []
    all_id = []

    # # some settings
    # percentagePedestriansRunning = 0.1      # how many pedestrians will run
    # percentagePedestriansCrossing = 0.1     # how many pedestrians will walk through the road
    
    world.set_pedestrians_seed(seedw)
    random.seed(seedw)

    # 1. take all the random locations to spawn
    spawn_points = []
    for i in range(number_of_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(world.get_blueprint_library().filter('walker.pedestrian.*'))

        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            print(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            print(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put together the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # # wait for a tick to ensure client receives the last transform of the walkers we have just created
    # if args.asynch or not synchronous_master:
    #     world.wait_for_tick()
    # else:
    #     world.tick()
    world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    return walkers_list, all_id

class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, world, vehicle, col_threshold=50):
        """Constructor method"""
        self.sensor = None
        # self.history = []
        self.vehicle = vehicle
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.vehicle)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
        self.collided = False
        self.col_threshold = col_threshold

    # def get_collision_history(self):
    #     """Gets the history of collisions"""
    #     history = collections.defaultdict(int)
    #     for frame, intensity in self.history:
    #         history[frame] += intensity
    #     return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        # actor_type = get_actor_display_name(event.other_actor) # collide with this actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        # self.history.append((event.frame, intensity))
        # if len(self.history) > 4000:
        #     self.history.pop(0)
        if intensity > self.col_threshold:
            self.collided = True


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, world, vehicle):
        """Constructor method"""
        self.sensor = None
        self.vehicle = vehicle
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
        self.invaded = False

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        self.invaded = True
