import atexit
import collections
import os
import random
import signal
import subprocess
import sys
import time
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import pygame

import carla
from carla import ColorConverter as cc
import numpy as np
import transforms3d.euler
import logging
import argparse

import global_planner
from CarlaSyncMode import CarlaSyncMode
import freemap_extraction as fme
from freemap import Freemap
import instruction_generation as ig
import util
from PIL import Image
import matplotlib.pyplot as plt

from run_experiment_xworld_rh import HUD, World, MiniMap, StateManager, spawn_cameras, KeyboardControl

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import copy
import carla_env_final
#from CarlaEnvSetup import setup

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import atexit
import collections
import os
import random
import signal
import subprocess
import sys
import time
#import gym
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from queue import Queue
import gymnasium as gym

import cv2

from utils_carmake import get_anno, get_half_anno
from gym import Wrapper

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time
from pathlib import Path
import signal
import json
import torch

import global_planner
from CarlaSyncMode import CarlaSyncMode
import freemap_extraction as fme
from freemap import Freemap
import instruction_generation as ig
import util
from PIL import Image
import matplotlib.pyplot as plt
#from gym.Wrappers import Monitor

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
    from pygame.locals import K_b
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# logging.set_verbosity(logging.DEBUG)

CONF = {"tm_seed": 2021,  # seed for traffic manager
        "car_lights_on": False,
        "percent_walking": 0,  # how many perdestrians will walk
        "percent_crossing": 0,  # how many pedestrians will cross road
        "percent_disabled": 0,
        "max_render_depth_in_meters": 100,
        "min_visible_vertices_for_render": 4,
        "num_vehicles": 200,
        "num_walkers": 0 # Changed from 50 to 0
        }

def setup(
    fps: int = 20,
    server_timestop: float = 10.0,
    client_timeout: float = 50.0,
):
    argparser = argparse.ArgumentParser(
        description="CARLA Pedestrian Control"
    )
    argparser.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="debug",
        help="print debug information"
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of host server (default: 127.0.0.1)"
    )
    argparser.add_argument(
        "-p","--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)"
    )
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="640x360",
        help="Window resolution (default: 640x360)"
    )
    argparser.add_argument(
        "--filter",
        metavar="PATTERN",
        default="walker.pedestrian.0001",
        help="actor filter (default: 'walker.pedestrian.0001')"
    )
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--world',
        default='Town10HD_Opt',
        type=str,
        help='World map')
    argparser.add_argument(
        '--destination_id',
        default=95,
        type=int,
        help='Destination point ID')
    argparser.add_argument(
        '--start_id',
        default=32,
        type=int,
        help='Start point ID')
    argparser.add_argument(
        '--weather',
        default='WetCloudySunset',
        type=str,
        help='Weather')
    argparser.add_argument(
        '--tm_port',
        default=6000,
        type=int,
        help='TrafficManager Port')

    argparser.add_argument(
        '--path',
        default="/data/kathakoli/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_32_95.npy",
        type=str,
        help='Global path file'
    )
    argparser.add_argument(
        '--log_dir',
        default="../logs/",
        type=str,
        help='Global log directory'
    )
    argparser.add_argument(
        '--repeat_action',
        help='Number of times action to be repeated'
    )
    argparser.add_argument(
        '--steps_per_episode',
        help='Steps'
    )
    argparser.add_argument(
        '--model_name',
        help='name of model when saving')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split("x")]
    port = args.port

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # logging.info('Starting server %s: %s'. args.host, args.port)
    env = os.environ.copy()
    # env["SDL_VIDEODRIVER"] = "offscreen"
    # env["SDL_HINT_CUDA_DEVICE"] = "0"
    env["CARLA_ROOT"]="/data2/kathakoli/carla/Unreal/CarlaUE4/Saved/StagedBuilds/LinuxNoEditor/"
    #env["CUDA_VISIBLE_DEVICES"]= "1"
    logging.debug("Inits a CARLA server at port={}".format(port))
    #CUDA_VISIBLE_DEVICES=3 ' + 
    server = subprocess.Popen(str(os.path.join(env["CARLA_ROOT"], "CarlaUE4.sh")), stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, shell=True)
    atexit.register(os.killpg, server.pid, signal.SIGKILL)
    time.sleep(server_timestop)

    print(__doc__)

    display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

    tag_set = list(fme.Tags.Hash2.values())
    path_finder = ig.PathFinder()
    world = None
    vehicles_list = []
    all_id_dict = {}

    # Connect client.
    logging.debug("Connects a CARLA client at port={}".format(port))
    
    # Setting up CARLA client
    client = carla.Client(args.host, args.port)
    client.set_timeout(client_timeout)
    hud = HUD(args.width, args.height)
    minimap = MiniMap(args.path)
    state_manager = StateManager(minimap, args.log_dir)
    print(args.world)
    client.load_world(args.world)

    client.get_world().unload_map_layer(carla.MapLayer.ParkedVehicles)
    client.get_world().unload_map_layer(carla.MapLayer.Props)
    client.get_world().unload_map_layer(carla.MapLayer.Decals)

    world = World(client.get_world(),hud, minimap, state_manager, args)
    print("World Loaded")

    if args.weather in util.WEATHER:
        weather = util.WEATHER[args.weather]
    else:
        weather = util.WEATHER['ClearNoon']
    world.world.set_weather(weather)
    print("Weather Initialized")

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)

    settings = world.world.get_settings()
    traffic_manager.set_synchronous_mode(True)

    if not settings.synchronous_mode:
        synchronous_master = True
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
    else:
        synchronous_master = False
    print("Synchronous: ", synchronous_master)
    world.world.apply_settings(settings)

    vehicles_list = util.spawn_vehicles(client, world.world, CONF['num_vehicles'], True, traffic_manager)

    if not util.spawn_walkers(client, world.world, CONF['num_walkers'], CONF['percent_disabled'], CONF['percent_walking'], CONF['percent_crossing'], all_id_dict):
        print("spawn_walkers failed")
        return
    
    # spawn dense
    dense_N = 0
    dense_percent_disabled = 0.0
    player_location = world.player.get_transform()
    loc_center = [player_location.location.x, player_location.location.y, player_location.location.z]

    controller = KeyboardControl(world)

    # print("loc_center: ", loc_center)

    if not util.spawn_walkers_dense(client, world.world, dense_N, loc_center,dense_percent_disabled, CONF['percent_walking'], CONF['percent_crossing'], all_id_dict):
        print("spawn_walkers_dense failed")
        return
    
    # cameras = spawn_cameras(world.world, world.player, world.destination_vehicle, args.width, args.height, 90)
    # state_manager.start_handler(world)

    all_actors = world.world.get_actors()
    all_vehicles = []
    all_peds = []
    for _a in all_actors:
        if 'vehicle' in _a.type_id:
            print(_a.type_id)
            all_vehicles.append(_a)

        if  _a.type_id.startswith('walker'):
            print(_a.type_id)
            all_peds.append(_a)
    
    return client, world, server, minimap, state_manager, traffic_manager, controller, display

# def reset(sync_mode, world,minimap,prev_x,prev_y):
#     count+=1
#     print("Inside reset",count)
#     frame_step = 0
#     l2_distance=0
#     route_completion=0
#     #self.distance_covered=0

#     if(count>1):
#         for camera in list(cameras.values()):
#             camera.stop()
#             camera.destroy()
#         world.restart()

#     cameras = {}
#     #640,360
#     #print(self.world.destination_vehicle)
#     cameras = spawn_cameras(world.world, world.player, world.destination_vehicle,640, 360, 90)
    
#     #prinprint(self.cameras)
#     sync_mode = CarlaSyncMode(world.world, *list(cameras.values()), fps=20)
#     try:
#         sync_mode.__enter__()  # Explicitly enter the context
#         #self.snapshot, self.eyelevel_rgb, self.eyelevel_ins, self.eyelevel_dep = self.sync_mode.tick(1.0)
#     finally:
#         sync_mode.__exit__() 
    
#     prev_x=-30
#     prev_y=57
#     #self.prev_time=time.time()

#     snapshot, eyelevel_rgb, eyelevel_ins, eyelevel_dep = sync_mode.tick(1.0)
#     eyelevel_rgb_array = util.get_image(eyelevel_rgb)
#     eyelevel_rgb_array = np.uint8(np.array(eyelevel_rgb_array))
#     eyelevel_rgb_array = cv2.resize(eyelevel_rgb_array,(640,360))
#     eyelevel_rgb_array = eyelevel_rgb_array[:, :, :3]
#     eyelevel_rgb_array=cv2.cvtColor(eyelevel_rgb_array, cv2.COLOR_RGB2BGR)
#     pedestrian_detected = 1 if _get_num_people(eyelevel_rgb_array,model_yolo) > 0 else 0
#     print("Collision history :",world.collision_sensor.get_collision_history())
#     collision_history = world.collision_sensor.get_collision_history()
#     collision_detected = 0
#     if len(collision_history) > 0:
#         if all(value==0.0 for value in collision_history.values()):
#             collision_detected = 0
#         else:
#             collision_detected = 1

#     # collision_detected=len(self.world.collision_sensor.get_collision_history())
#     distance_covered=np.linalg.norm(np.array([minimap.player_pos.x,minimap.player_pos.y]) - np.array([prev_x,prev_y]))
#     #distance=self.time_covered*self.speed
#     #obs= collections.OrderedDict([('pedestrian_detected', pedestrian_detected), ('collision_detected', collision_detected), ('distance_covered', distance_covered)])
    
    
#     print("Inside Setup", self.controller._control)
    
#     info = dict()
    
#     print("Starting Game Loop")
#     state_manager.start_handler(world)
    
#     world.tick(clock)

#     #cv2.imwrite('image.jpg',self.eyelevel_rgb_array)
#     #self.eyelevel_rgb_array.save_to_disk('_out/image.jpg')

#     return pedestrian_detected, collision_detected, distance_covered,sync_mode, prev_x, prev_y

# # def _get_obs(sync_mode):
# #     snapshot, eyelevel_rgb, eyelevel_ins, eyelevel_dep = sync_mode.tick(1.0)
# #     eyelevel_rgb_array = util.get_image(eyelevel_rgb)
# #     eyelevel_rgb_array = np.uint8(np.array(eyelevel_rgb_array))
# #     eyelevel_rgb_array = cv2.resize(eyelevel_rgb_array,(640,360))
# #     eyelevel_rgb_array = eyelevel_rgb_array[:, :, :3]
# #     eyelevel_rgb_array=cv2.cvtColor(eyelevel_rgb_array, cv2.COLOR_RGB2BGR)
# #     pedestrian_detected = 1 if _get_num_people(eyelevel_rgb_array,model_yolo) > 0 else 0
# #     print("Collision history :",world.collision_sensor.get_collision_history())
# #     collision_history = world.collision_sensor.get_collision_history()
# #     collision_detected = 0
# #     if len(collision_history) > 0:
# #         if all(value==0.0 for value in collision_history.values()):
# #             collision_detected = 0
# #         else:
# #             collision_detected = 1

# #     # collision_detected=len(self.world.collision_sensor.get_collision_history())
# #     distance_covered=np.linalg.norm(np.array([minimap.player_pos.x,self.minimap.player_pos.y]) - np.array([prev_x,prev_y]))
# #     #distance=self.time_covered*self.speed
# #     return collections.OrderedDict([('pedestrian_detected', pedestrian_detected), ('collision_detected', collision_detected), ('distance_covered', distance_covered)])

def render(world,display, mode='human'):
    # TODO: clean this
    # TODO: change the width and height to compat with the preview cam config

    world.render(display)
    pygame.display.flip()

def main():
    pygame.init()
    pygame.font.init()
    world = None

    tag_set = list(fme.Tags.Hash2.values())
    path_finder = ig.PathFinder()
    vehicles_list = []
    all_id_dict = {}
    client, world, server, minimap, state_manager, traffic_manager,controller,display=setup()
    episode_rewards=[]
    total_reward = 0
    prev_x=-30
    prev_y=57
    l2_distance=0
    distance_covered=0
    im_width = args['width']
    im_height = args['height']
    repeat_action = args['repeat_action']

    steps_per_episode =int(args['steps_per_episode'])
    model_yolo = torch.hub.load("ultralytics/yolov5","yolov5s",trust_repo=True,device=1)
    all_id_dict=all_id_dict

    clock = pygame.time.Clock()
    count=0
    #self.collision_count=0
    task_metrics=[]
    count_epi=0
    for num_steps in range(10):
        count+=1
        print("Inside reset",count)
        frame_step = 0
        l2_distance=0
        route_completion=0
        #self.distance_covered=0

        if(count>1):
            for camera in list(cameras.values()):
                camera.stop()
                camera.destroy()
            world.restart()

        cameras = {}
        #640,360
        #print(self.world.destination_vehicle)
        cameras = spawn_cameras(world.world, world.player, world.destination_vehicle,640, 360, 90)
        
        #prinprint(self.cameras)
        sync_mode = CarlaSyncMode(world.world, *list(cameras.values()), fps=20)
        try:
            sync_mode.__enter__()  # Explicitly enter the context
            #self.snapshot, self.eyelevel_rgb, self.eyelevel_ins, self.eyelevel_dep = self.sync_mode.tick(1.0)
        finally:
            sync_mode.__exit__() 
        
        prev_x=-30
        prev_y=57
        #self.prev_time=time.time()

        snapshot, eyelevel_rgb, eyelevel_ins, eyelevel_dep = sync_mode.tick(None)
        eyelevel_rgb_array = util.get_image(eyelevel_rgb)
        eyelevel_rgb_array = np.uint8(np.array(eyelevel_rgb_array))
        eyelevel_rgb_array = cv2.resize(eyelevel_rgb_array,(640,360))
        eyelevel_rgb_array = eyelevel_rgb_array[:, :, :3]
        eyelevel_rgb_array=cv2.cvtColor(eyelevel_rgb_array, cv2.COLOR_RGB2BGR)
        pedestrian_detected = 1 if _get_num_people(eyelevel_rgb_array,model_yolo) > 0 else 0
        print("Collision history :",world.collision_sensor.get_collision_history())
        collision_history = self.world.collision_sensor.get_collision_history()
        collision_detected = 0.0
        if len(collision_history.keys()) > 0:
            # if all(value==0.0 for value in collision_history.values()):
            #     collision_detected = 0
            # else:
            collision_ids = self.world.collision_sensor.get_collision_ids()
            if any("walker" in value for value in collision_ids.values()):
                collision_detected = 1.0

        # collision_detected=len(self.world.collision_sensor.get_collision_history())
        l2_distance=np.linalg.norm(np.array([minimap.player_pos.x,minimap.player_pos.y]) - np.array([prev_x,prev_y]))
        #distance=self.time_covered*self.speed
        #obs= collections.OrderedDict([('pedestrian_detected', pedestrian_detected), ('collision_detected', collision_detected), ('distance_covered', distance_covered)])
        
        
        print("Inside Setup", self.controller._control)
        
        info = dict()
        
        print("Starting Game Loop")
        state_manager.start_handler(world)
        
        world.tick(clock)
        #distance_covered=0
        episode_rewards.append(total_reward)
        total_reward = 0
        for steps in range(1000):
            l2_distance=0
            world.tick(clock)
            render(world,display)
                
            frame_step += 1
            info = dict()
            done =truncated=False
            reward = 0
            snapshot, eyelevel_rgb, eyelevel_ins, eyelevel_dep = sync_mode.tick(None)
            eyelevel_rgb_array = util.get_image(eyelevel_rgb)
            eyelevel_rgb_array = np.uint8(np.array(eyelevel_rgb_array))
            eyelevel_rgb_array = cv2.resize(eyelevel_rgb_array,(640,360))
            eyelevel_rgb_array = eyelevel_rgb_array[:, :, :3]
            eyelevel_rgb_array=cv2.cvtColor(eyelevel_rgb_array, cv2.COLOR_RGB2BGR)
            pedestrian_detected = 1 if _get_num_people(eyelevel_rgb_array,model_yolo) > 0 else 0
            
            #distance=self.time_covered*self.speed
            #return collections.OrderedDict([('pedestrian_detected', pedestrian_detected), ('collision_detected', collision_detected), ('distance_covered', distance_covered)])

            if pedestrian_detected==1:
                print("Running local policy")
                controller._control.speed = 0.0
                world.player.apply_control(controller._control)# Local Policy-STOP
                print("Inside local policy", controller._control)
            else:
                print("Running global policy")
                controller._control.speed = 1.5
                controller._control.direction=_run_global_policy()
                world.player.apply_control(controller._control)
                print("Inside global policy", controller._control)

            print("Collision history :",world.collision_sensor.get_collision_history())
            collision_history = self.world.collision_sensor.get_collision_history()
            collision_detected = 0.0
            if len(collision_history.keys()) > 0:
                # if all(value==0.0 for value in collision_history.values()):
                #     collision_detected = 0
                # else:
                collision_ids = self.world.collision_sensor.get_collision_ids()
                if any("walker" in value for value in collision_ids.values()):
                    collision_detected = 1.0

            # collision_detected=len(self.world.collision_sensor.get_collision_history())
            l2_distance=np.linalg.norm(np.array([minimap.player_pos.x,minimap.player_pos.y]) - np.array([prev_x,prev_y]))
            cv2.imshow("walker camera View", eyelevel_rgb_array)
            cv2.waitKey(1)
            prev_x=minimap.player_pos.x
            prev_y=minimap.player_pos.y

            route_completion+=l2_distance
            reward+=100*(l2_distance)
            print("reward from route completion",(reward))
            if collision_detected!= 0:
            # if (len(self.collision_hist)-1) != 0:
                print("Collision History",len(world.collision_sensor.get_collision_history()))
                # print("Collision History",self.collision_hist)
                # done = True
                truncated=True
                reward += -3500
            if state_manager.end_state:
                # end experiment
                done = True
                #reward += 100     

            
            #self.prev_time=time.time()
            
            # print("Steps per episode",self.frame_step)
            # if self.frame_step >= self.steps_per_episode:
            #     done = True
            total_reward += reward
            if done or truncated:
                count_epi+=1
                print("Env lasts {} steps, restarting ... ".format(frame_step))
                print("Task Metrics:")
                print("Collision Count:",len(world.collision_sensor.get_collision_history()))
                print("Route Completion",(route_completion))
                #print("Energy",self.energy,"mJ")
                task_metrics.append([count_epi,len(world.collision_sensor.get_collision_history()),route_completion])
                #print("Energy",
                #self.collision_count+=1

                #obs, rew, done,truncated, info= self._step(action)
                break
                


if __name__ == '__main__':

    main()

