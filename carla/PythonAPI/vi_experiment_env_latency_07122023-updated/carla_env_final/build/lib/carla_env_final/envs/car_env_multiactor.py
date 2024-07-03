"""
Welcome to CARLA MultiActor Environment, based off https://github.com/johnMinelli/carla-gym/. 

This environment has been adapted for X-World / RoboPolicy Cloud Minimal Systems.
"""



from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import traceback
from copy import deepcopy

sys.path.append('data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/carla_env_final/carla_env_final/envs/')

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
import subprocess
#import gym
from typing import Any
from typing import List
import gymnasium as gym
from gymnasium import spaces #.spaces import Box, Dict

import cv2

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
import GPUtil
import socket

import global_planner
from CarlaSyncMode import CarlaSyncMode, CarlaSyncMultiMode
import util

from gymnasium.utils import EzPickle
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import ActionDict

from libcarla.command import SpawnActor as SpawnActor

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


# ==============================================================================
# -- Global CONF ---------------------------------------------------------------
# ==============================================================================

CONF = {"tm_seed": 2021,  # seed for traffic manager
        "car_lights_on": False,
        "percent_walking": 0,  # how many perdestrians will walk
        "percent_crossing": 0,  # how many pedestrians will cross road
        "percent_disabled": 0,
        "max_render_depth_in_meters": 100,
        "min_visible_vertices_for_render": 4,
        "num_vehicles": 200,
        "num_walkers": 50
        }

LOG_DIR = os.path.join(os.getcwd(), "logs")
live_carla_processes = dict()
logger = logging.getLogger(__name__)

def cleanup():
    print("Killing live carla processes", live_carla_processes)
    for container_name, pgid  in live_carla_processes.items():
        # for Linux
        os.killpg(pgid, signal.SIGKILL)

        container_stop = subprocess.Popen(
            (
                f"docker stop {container_name}"
            ),
            shell=True
        )
        time.sleep(11)

    live_carla_processes.clear()

def termination_cleanup(*_):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, termination_cleanup)
signal.signal(signal.SIGINT, termination_cleanup)
atexit.register(cleanup)


def env(**kwargs):
    env = CarlaMultiActorEnvPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class CarlaMultiActorEnv(gym.Env):
    """
        Carla Multi-Actor Gym environment, based off Carla-Gym parallelization
        For RoboPolicy/Cloud Minimal Systems setup
    """

    def __init__(
        self,
        train: bool = True,
        steps_per_episode: int = 1500,
        img_width: int = 640,
        img_height: int = 360,
        fixed_delta_seconds: float = 0.05,
        fps: int = 20,
        server_timeout: float = 5.0,
        client_timeout: float = 30.0,
        host: str = "localhost",
        paths: List[str] = [
                "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_04_57_1.npy",
                "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_21_86_1.npy",
                "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_32_95.npy",
                "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_61_11_1.npy",
                "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_131_46_1.npy"
                ],
        world: str = "Town10HD_Opt",
        actor_filter: str = "walker.pedestrian.0001",
        weather: str = "WetCloudySunset",
        logs_dir: str = "../logs/",
        rolename: str = "hero",
        gamma: float = 2.2,
        spawn_pts_file: str = "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/start.npy",
        destination_pts_file: str = "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/end.npy",
    ):
        try:
            spawn_pts = np.load(spawn_pts_file)
            destination_pts = np.load(destination_pts_file)
        except:
            raise Exception("Error in loading the spawn and destination point files, please check")
    
        
        assert len(paths)== len(spawn_pts) == len(destination_pts), f"The number of paths {len(paths)}, spawn points {len(spawn_pts)} and destination points {len(destination_pts)} don't match"
        self.metadata = {"render_modes": "human", "render_fps": fps}
        
        self.train = train

        pygame.init()
        pygame.font.init()

        # CARLA related variables
        self.host=host
        self.server_timeout = server_timeout
        self.client_timeout = client_timeout
        self.path_files = paths
        self.world_name = world
        self.world_weather = weather
        self.actor_filter = actor_filter
        self.logs_dir = logs_dir
        self.fixed_delta_seconds = fixed_delta_seconds
        
        # CARLA Variables set/required to start server and related stuff
        self.server_name = "carla_train" if self.train else "carla_eval"
        
        self.actor_ids = sorted([rolename+f"_{i}" for i in range(len(self.path_files))])
        self.gamma = gamma
        self.spawn_points = spawn_pts
        self.destination_pts = destination_pts
        self.client = None
        self.world = None
        self.server_port = None
        self.server = None
        self.state_manager = {}
        self.minimap = {}
        self.traffic_manager = None
        self.controller = {}
        self.display = {}
        self.sparse_id_dict = None
        self.dense_id_dict = None
        self.vehicles_list = None
        self.clock = pygame.time.Clock()

        # Initial condition to start server, client, etc.
        self.do_initial_reset = True

        # CARLA Step/Reset variables
        self.sync_mode = None
        self.cameras = {}
        self.dones = {}
        self._terminations = {a: False for a in self.actor_ids}
        self._truncations = {a: False for a in self.actor_ids}
        self.info = {a: {} for a in self.actor_ids}
        
        
        #Render stuff
        self.img_width = img_width
        self.img_height = img_height

        
        # Environment local variables
        self.steps_per_episode = steps_per_episode
        self.frame_step = collections.defaultdict(int)
        self.episode_count = collections.defaultdict(int)

        # Step/Obs/Reset variables
        self.q = collections.defaultdict(int)
        self.sampled_latency = collections.defaultdict(int)
        self.expert_policy_list = collections.defaultdict(lambda: np.array([[0, 0, 0] for _ in range(4)]))
        self.rotation_new = collections.defaultdict(float)
        self.speed_new = collections.defaultdict(float)


        self.action_spaces = spaces.Dict(
            {
                actor_id: spaces.Box(low=np.array([[-3.14,0.0]]), high=np.array([[-3.14,0.0]]), shape=(1,2), dtype=np.float64)
                for actor_id in self.actor_ids
            }
        )

        # print(self.action_spaces)
        # print(isinstance(self.action_space('hero_0'),spaces.Space))
        # Observation space
        self.observation_spaces = spaces.Dict(
            {
                actor_id: spaces.Box(low=np.array([[-3.14,0.0,0.0] for _ in range(4)]), high=np.array([[-3.14,0.0,0.0] for _ in range(4)]), shape=(4,3), dtype=np.float64)
                for actor_id in self.actor_ids
            }
        )
    

    # @property
    def observation_space(self, actor_id: str):
        return self.observation_spaces[actor_id]
    
    # @property
    def action_space(self, actor_id: str):
        return self.action_spaces[actor_id]

    
    @property
    def num_walkers(self): # Gym Accessor
        """Return number of ego walkers"""
        return len(self.actor_ids)


    @staticmethod
    def _get_tcp_port(port: int = 0):
        """Get a free TCP port number.

        Args:
          port (Optional[int]): Port number. When set to `0`, it will be assigned a free port dynamically.

        Returns:
            A port number requested if free, otherwise an unhandled exception would be thrown.
        """
        s = socket.socket()
        s.bind(("", port))
        server_port = s.getsockname()[1]
        s.close()
        return server_port
    
    def _init_server(self):
        self.server_port = self._get_tcp_port()
        gpus = GPUtil.getGPUs()
        log_file = os.path.join(LOG_DIR, "docker_server_"+str(self.server_port)+".log")
        logger.info(
            f"CARLA server port: {self.server_port}\n"
        )

        if gpus is not None and len(gpus)>0:
            print("Initializing server")
            try:
                min_index = random.randint(0, len(gpus) - 1)
                for i, gpu in enumerate(gpus):
                    if gpu.memoryFree > gpus[min_index].memoryFree:
                        min_index = i
                
                self.server = subprocess.Popen(
                    (
                        f"docker run --rm --privileged --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES={min_index} --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw --name {self.server_name} carlasim/carla:0.9.13 /bin/bash ./CarlaUE4.sh -world-port -world-port={self.server_port} -RenderOffScreen -quality-level=Epic -nosound"
                    ),
                    shell=True,
                    preexec_fn=os.setsid,
                    creationflags=0,
                    stdout=open(log_file, "w")
                )
                time.sleep(self.server_timeout)

                if self.server.errors is not None:
                    raise Exception(
                        f"Subprocess returned code {self.server.returncode},"
                        f"Output: {self.server.stdout}, Error: {self.server.stderr}"
                        f"Args: {self.server.args}"
                    )
                
                print(f"CARLA Docker server running on port {self.server_port} and GPU {min_index}")

            except Exception as e:
                print(e)
        
        live_carla_processes.update({self.server_name:os.getpgid(self.server.pid)})


        # starting client
        self.client=None
        while self.client is None:
            try:
                self.client = carla.Client(self.host, self.server_port)
                time.sleep(2)
                self.client.set_timeout(self.client_timeout)
                print("Client connected to server")
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    raise Exception("Could not connect to Carla server:", re)
                self._client = None
        
        try:
            hud = {}
            self.display = {}
            for actor_id, path_file in zip(self.actor_ids, self.path_files):
                self.display[actor_id] = pygame.display.set_mode((self.img_width, self.img_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                # print("Display ready")
                hud[actor_id] = HUD(self.img_width, self.img_height, actor_id)
                # print("HUD ready")
                self.minimap[actor_id] = MiniMap(path_file, actor_id)
                # print("MiniMap ready")
                self.state_manager[actor_id] = StateManager(self.minimap[actor_id], self.logs_dir)
                # print("State Manager ready")
            
            # hud = HUD(self.img_width, self.img_height)
            # self.minimap = MiniMap(self.path)
            # self.state_manager = StateManager(self.minimap, self.logs_dir)

            # Load world and unload map layers
            self.client.load_world(self.world_name)
            self.client.get_world().unload_map_layer(carla.MapLayer.ParkedVehicles)
            self.client.get_world().unload_map_layer(carla.MapLayer.Props)
            self.client.get_world().unload_map_layer(carla.MapLayer.Decals)

            
            self.world = World(self.client.get_world(), hud, self.minimap, self.state_manager, self.gamma, self.actor_ids, self.actor_filter, self.spawn_points)

            if self.world_weather in util.WEATHER:
                weather = util.WEATHER.get(self.world_weather)
            else:
                weather = util.WEATHER.get('ClearNoon')
            
            self.world.world.set_weather(weather)
            print(f"Weather initialized: {weather}")

            tm_port = self._get_tcp_port()
            self.traffic_manager = self.client.get_trafficmanager(tm_port)
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(70.0)

            settings = self.world.world.get_settings()
            self.traffic_manager.set_synchronous_mode(True)

            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.fixed_delta_seconds
            else:
                synchronous_master = False
            print("Synchronous: ", synchronous_master)
            self.world.world.apply_settings(settings)

            self.vehicles_list = []
            self.vehicles_list = util.spawn_vehicles(self.client, self.world.world, CONF['num_vehicles'], True, self.traffic_manager)
            # pedestrian_seed = 42
            # self.world.world.set_pedestrians_seed(pedestrian_seed)  
            
            self.sparse_id_dict = {}
            if not util.spawn_walkers(self.client, self.world.world, 30, 0, 0.9, 0.2, self.sparse_id_dict):
                print("spawn_walkers failed")
            
            self.dense_id_dict = {}
            self.controller = {}
            # print(self.world.player.items())
            # print(self.destination_pts)
            for actor_id, actor, destination_pt in zip(self.world.player.keys(), self.world.player.values(), self.destination_pts):
                dense_id_dict = {}
                player_location = actor.get_transform()
                loc_center = [player_location.location.x, player_location.location.y, player_location.location.z]
                spawn_walkers_check = util.spawn_walkers_dense(self.client, self.world.world, 30, loc_center, 0,0.9,0.2, dense_id_dict, actor_id, destination_pt, self.train)
                if not spawn_walkers_check:
                    print("spawn_walkers_dense failed")
                
                self.dense_id_dict[actor_id] = dense_id_dict

                self.controller[actor_id] = KeyboardControl(self.world, actor_id)
            # print("After spawning dense walkers: ",self.dense_id_dict.keys())

        except Exception as error:
            logger.error(error)
            raise error

    

    # Have to fix resets before working on step
    def _reset(self, reset_actors: bool = True):
        if reset_actors:
            self.dones = {"__all__": False}
            self._terminations = {}
            self._truncations = {}
            for actor_id in self.actor_ids:
                for camera in list(self.cameras[actor_id].values()):
                    camera.stop()
                    camera.destroy()
                
                for i in range(len(self.dense_id_dict[actor_id]['controllers'])):
                    controller = self.world.world.get_actor(self.dense_id_dict[actor_id]['controllers'][i])
                    controller.stop()
                
                self.expert_policy_list[actor_id] = np.array([[0, 0, 0] for _ in range(4)])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.dense_id_dict[actor_id]['controllers']])
                self.world.tick(self.clock)

                transforms = [carla.command.ApplyWalkerState(act_id, x[0],0.0) for act_id, x in zip(self.dense_id_dict[actor_id]['walkers'], self.dense_id_dict[actor_id]['locations'])]

                self.dones[actor_id] = False
                self._terminations[actor_id] = False
                self._truncations[actor_id] = False

                self.client.apply_batch(transforms)

                self.world.tick(self.clock)

                
            for actor_id in self.actor_ids:
                batch = []
                controller_list = []
                controller_bp = self.world.world.get_blueprint_library().find('controller.ai.walker')
                for walker in self.dense_id_dict[actor_id]['walkers']:
                    batch.append(SpawnActor(controller_bp, carla.Transform(), walker))
                for response in self.client.apply_batch_sync(batch, True):
                    if response.error:
                        print(response.error)
                    else:
                        controller_list.append(response.actor_id)
                
                self.dense_id_dict[actor_id].update({'controllers': controller_list})

                for i in range(len(self.dense_id_dict[actor_id]['controllers'])):
                    location = self.dense_id_dict[actor_id]['locations'][i]

                    controller = self.world.world.get_actor(self.dense_id_dict[actor_id]['controllers'][i])
                    controller.start()
                    controller.go_to_location(location[1])
                
            self.world.restart()
        
        else:
            self.dones["__all__"] = False



        self.cameras = {}
        for actor_id in self.actor_ids:
            self.cameras[actor_id] = spawn_cameras(self.world.world, self.world.player[actor_id], 1024, 576, 120)
            # self.sync_mode[actor_id] = CarlaSyncMode(self.world.world, *list(self.cameras[actor_id].values()), fps=20)
            # try:
            #     self.sync_mode[actor_id].__enter__()
            # finally:
            #     self.sync_mode[actor_id].__exit__()
        
        self.sync_mode = CarlaSyncMultiMode(self.world.world, self.cameras, fps=20, actor_ids = self.actor_ids)
        try:
            self.sync_mode.__enter__()
        finally:
            self.sync_mode.__exit__()

        
        self.world.tick(self.clock)

        
        raw_obs = self._get_obs()

        self.prev_obs = {}
        info = collections.defaultdict(dict)
        for actor_id  in self.actor_ids:
            self.prev_obs[actor_id] = raw_obs[actor_id][0]
        
        return self.prev_obs



    def reset(self, seed = None, options = None):
        if self.do_initial_reset:
            self._init_server()
            self.do_initial_reset = False
            self._reset(reset_actors=False)
        else:
            self._reset()


    def _run_global_policy(self, actor_id):
        player_pos = self.world.player[actor_id].get_transform().location
        
        local_goal = self.minimap[actor_id].planner.get_next_goal(
            pos=[player_pos.x, player_pos.y], preview_s=5)
        local_goal = np.array(local_goal)[None, :]
        
        ego_world_pos = np.array([[player_pos.x, player_pos.y]])

        rotation = math.atan2(local_goal[0][1]-ego_world_pos[0][1], local_goal[0][0]-ego_world_pos[0][0])
        
        return rotation

    
    def _get_actor_direction(self, actor, player_pos):
        actor_pos = actor.get_location()
        rotation = math.atan2(actor_pos.y-player_pos.y,actor_pos.x-player_pos.x)
        if abs(rotation) > (math.pi/4):
            return None
        return carla.Vector3D(math.cos(rotation),math.sin(rotation),0.0)
    
    def _get_actor_distance(self, actor, player_pos):
        return actor.get_location().distance(player_pos)


    def expert_policy(self,pedestrian_detected, actor_id):
        self.q[actor_id]+=1
        if self.q[actor_id]<self.sampled_latency[actor_id]:
            self.expert_policy_list[actor_id] = np.vstack((self.expert_policy_list[actor_id], np.array([self.rotation_new[actor_id], self.speed_new[actor_id],self.q[actor_id]])))  
        else:
            if pedestrian_detected==0:
                self.speed_new[actor_id] = 1.5
                self.rotation_new[actor_id]=self._run_global_policy(actor_id)
            else:
                self.speed_new[actor_id] = 0
                self.rotation_new[actor_id]=0
            self.sampled_latency[actor_id]= int(np.random.normal(5, 2.5, 1))
            # print(type(self.sampled_latency),self.sampled_latency,"Latency sampling")
            self.q[actor_id]=0 
            self.expert_policy_list[actor_id] = np.vstack((self.expert_policy_list[actor_id], np.array([self.rotation_new[actor_id], self.speed_new[actor_id],self.q[actor_id]])))       
        return self.expert_policy_list[actor_id][-4:]

    def get_reward(self, actor_id, collision, action, geodesic, dir_r):
        """
        Calculate reward for 
        """
        if collision !=0 and self.world.player[actor_id].get_velocity().length() > 1e-5:
            return -20.0
        else:
            extreme_action0_rw = 1.0 if abs(action[0][0]/3.14) < 0.97 else 0.0
            extreme_action1_rw = 1.0 if abs(action[0][1]/2)< 0.97 else 0.0
            extreme_actions_reward = (extreme_action0_rw*extreme_action1_rw)**.5
            #print("Geodesic",geodesic)
            geodesic_rw = 1.0 - math.tanh(geodesic)
            #print(geodesic_rw)
            #speed=(action[0][1])/2x
            speed=self.world.player[actor_id].get_velocity().length()
            #print(geodesic_rw)
            #return geodesic_rw
            #print((geodesic_rw*speed*extreme_actions_reward)**(1.0/3.0))
            return (geodesic_rw*speed*extreme_actions_reward*dir_r)**(1.0/3.0)

    def _get_obs(self):
        if self.train:
            self.sync_mode.tick(None)
            # for actor_id in self.actor_ids:
            #     self.snapshot, self.eyelevel_rgb, self.eyelevel_ins, self.eyelevel_dep = self.tick_data[actor_id]
            #     self.eyelevel_rgb_array = util.get_image(self.eyelevel_rgb)
            #     self.eyelevel_rgb_array = np.uint8(np.array(self.eyelevel_rgb_array))
            #     self.eyelevel_rgb_array = cv2.resize(self.eyelevel_rgb_array,(640,360))
            #     self.eyelevel_rgb_array = self.eyelevel_rgb_array[:, :, :3]
            #     self.eyelevel_rgb_array=cv2.cvtColor(self.eyelevel_rgb_array, cv2.COLOR_RGB2BGR)
            #     cv2.imshow(f"walker camera View - {actor_id}", self.eyelevel_rgb_array)
            #     cv2.waitKey(1)
        else:
            self.tick_data = self.sync_mode.tick(None)
            for actor_id in self.actor_ids:
                self.snapshot, self.eyelevel_rgb, self.eyelevel_ins, self.eyelevel_dep = self.tick_data[actor_id]
                self.eyelevel_rgb_array = util.get_image(self.eyelevel_rgb)
                self.eyelevel_rgb_array = np.uint8(np.array(self.eyelevel_rgb_array))
                self.eyelevel_rgb_array = cv2.resize(self.eyelevel_rgb_array,(640,360))
                self.eyelevel_rgb_array = self.eyelevel_rgb_array[:, :, :3]
                self.eyelevel_rgb_array=cv2.cvtColor(self.eyelevel_rgb_array, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"walker camera View - {actor_id}", self.eyelevel_rgb_array)
                cv2.waitKey(1)

        self.ped_count = self.col_count = collections.defaultdict(int)
        self.pedestrian_list=self.world.world.get_actors().filter("walker.*")
        # for actor_id in self.actor_ids:
        #     self.pedestrian_list.remove(self.world.player[actor_id])
        # self.pedestrian_list.remove(self.world.player)
        
        obs = {}
        for actor_id in self.actor_ids:
            player_pos = self.world.player[actor_id].get_location()
            ped_distances = []
            for i in self.pedestrian_list:
                #print(i)
                #ped_distances.append(self._get_actor_distance(i, player_pos))
                ped_distances.append(self._get_actor_distance(i, player_pos))
                if self._get_actor_distance(i, player_pos)<4 and self._get_actor_direction(i, player_pos)!=None:
                    self.ped_count[actor_id]+=1
            #print(sorted(ped_distances))
            # print(np.sort(col_count))
            pedestrian_detected = 1.0 if self.ped_count[actor_id] > 1 else 0.0
            policy_list=self.expert_policy(pedestrian_detected, actor_id)
            for i in self.pedestrian_list:
                #print(i)
                #ped_distances.append(self._get_actor_distance(i, player_pos)
                if self._get_actor_distance(i, player_pos)<1 and self._get_actor_direction(i, player_pos)!=None:
                    # print(self._get_actor_distance(i, player_pos),'Hi')
                    self.col_count[actor_id]+=1
            collision_detected = 1.0 if self.col_count[actor_id] > 1 else 0.0
            #print("Pedestrian Count", self.ped_count)
            
            p=self.minimap[actor_id].planner.get_closest_point(np.array([self.minimap[actor_id].player_pos.x,self.minimap[actor_id].player_pos.y]))
            #print(p)
            pathxclose,pathyclose=self.minimap[actor_id].planner.path_pts[p,:2]
            pathx,pathy=self.minimap[actor_id].planner.get_next_goal(pos=[self.minimap[actor_id].player_pos.x,self.minimap[actor_id].player_pos.y],preview_s=2)
            # print(self.minimap.player_pos.x,self.minimap.player_pos.y)
            # print(pathx,pathy)
            path_next_x,path_next_y=self.minimap[actor_id].planner.get_next_goal(pos=[pathxclose,pathyclose],preview_s=2)
            #print(path_next_x,path_next_y)
            distance_covered=np.linalg.norm(np.array([pathxclose,pathyclose])-np.array([self.minimap[actor_id].player_pos.x,self.minimap[actor_id].player_pos.y]))
            ped_vector=carla.Location(pathx,pathy,0)-carla.Location(self.minimap[actor_id].player_pos.x,self.minimap[actor_id].player_pos.y,0)
            #print(ped_vector)
            path_vector=carla.Location(path_next_x,path_next_y,0)-carla.Location(pathxclose,pathyclose,0)
            #print(path_vector)
            angle=path_vector.get_vector_angle(ped_vector)
            # print("Angle",angle)
            if angle>=0 and angle<=3.14: 
                r=1
            else:
                r=-1
            distance_to_nearest_pedestrian = sorted(ped_distances)[1] if len(ped_distances) > 1 else 0.0
            # print(type(distance_covered))
            #distance=self.time_covered*self.speed
            #print(policy_list,"Expert policy")
            obs[actor_id] = (policy_list,np.array([pedestrian_detected,collision_detected,distance_covered,r,distance_to_nearest_pedestrian], dtype=np.float64))
        return obs



    def _step(self, actions: dict):
        """
        Run actual step for actor in CARLA

        Args:
        actor_id (str): Actor identifier
        action: Action to be executed for the actor
        """
        l2_distance = collections.defaultdict(float)
        reward = collections.defaultdict(float)
        done = truncated = collections.defaultdict(bool)
        
        self.world.tick(self.clock)
        self.render()
        
        for actor_id, action in actions.items():
            # if not self.dones[actor_id]:
            x_dir = math.cos(action[0][0])
            y_dir = math.sin(action[0][0])
            self.controller[actor_id]._control.direction = carla.Vector3D(x_dir, y_dir, 0.0)
            self.controller[actor_id]._control.speed = action[0][1]
            self.world.player[actor_id].apply_control(self.controller[actor_id]._control)
        
        obs_to_use = self._get_obs()

        obs = {}
        for actor_id, action in actions.items():
            # if not self.dones[actor_id]:
            self.frame_step[actor_id] += 1
            l2_distance[actor_id] = obs_to_use[actor_id][1][2]
            reward[actor_id] = self.get_reward(actor_id, obs_to_use[actor_id][1][1], action, obs_to_use[actor_id][1][2],obs_to_use[actor_id][1][3])
            if self.world.player[actor_id].get_velocity().length() > 1e-5:
                if obs_to_use[actor_id][1][1] != 0.0:
                    truncated[actor_id] = True
            
            if self.state_manager[actor_id].end_state:
                truncated[actor_id] = True
            
            if self.frame_step[actor_id] >= self.steps_per_episode:
                truncated[actor_id] = True
            
            if done or truncated:
                self.episode_count[actor_id] += 1
            
            obs[actor_id] = obs_to_use[actor_id][0]
        
        self.prev_obs = obs

        print(self.frame_step)
        
        return self.prev_obs, reward, done, truncated, self.info






    def step(self, actions: dict):
        """
        Execute one environment step for actors.

        Args:
        actions (dict): Actions to be executed for each actor. 
        """
        if (not self.server) or (not self.client):
            raise RuntimeError("Cannot call step(...) before calling reset(...) first")

        assert len(self.world.player), "No actors exist in environment, something has gone wrong."

        if not isinstance(actions, dict):
            raise ValueError(f"`step(actions)` expected dict of actions. Got {type(actions)}")

        try:
            obs_dict = {}
            reward_dict = {}
            info_dict = {}


            obs, reward, done, truncated, info = self._step(actions)

            for actor_id, _ in actions.items():
                obs_dict[actor_id] = obs[actor_id]
                reward_dict[actor_id] = reward[actor_id]
                if not self.dones.get(actor_id, False):
                    self.dones[actor_id] = done[actor_id] or truncated[actor_id]
                    self._terminations[actor_id] = done[actor_id]
                    self._truncations[actor_id] = truncated[actor_id]
                info_dict[actor_id] = info[actor_id]
            
            self.dones["__all__"] = sum(self.dones.values()) >= self.num_walkers
            self.render()

            return obs_dict, reward_dict, self._terminations, self._truncations, info_dict

        except Exception:
            print("Error during step, terminating episode early.", traceback.format_exc())
            self._stop_server()
    
    def render(self):
        self.world.render(self.display)
        pygame.display.flip()


    def _stop_server(self):
        if self.server:
            pgid = os.getpgid(self.server.pid)
            os.killpg(pgid, signal.SIGKILL)
            del live_carla_processes[self.server_name]

            self.server_port = None
            self.server = None

            server_close = subprocess.Popen(
                (
                    f"docker stop {self.server_name}"
                ),
                shell=True
            )
            time.sleep(self.server_timeout*2.5)


    def close(self):
        print("Closing environment")
        if not self.do_initial_reset:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

            for _id in self.sparse_id_dict.get('controllers'):
                self.world.world.get_actor(_id).stop()
            
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sparse_id_dict['controllers']])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sparse_id_dict['walkers']])

            # print(self.dense_id_dict.keys())
            for actor_id in self.actor_ids:
                for _id in self.dense_id_dict[actor_id].get('controllers'):
                    self.world.world.get_actor(_id).stop()
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.dense_id_dict[actor_id]['controllers']])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.dense_id_dict[actor_id]['walkers']])

            for actor_id in self.actor_ids:
                for camera in list(self.cameras[actor_id].values()):
                    camera.stop()
                    camera.destroy()
            
            if self.world is not None:
                self.world.destroy()

            self._stop_server()











def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

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


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        # self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        # self._storm = Storm(weather.precipitation)

    def tick(self):
        # self._sun.tick(delta_seconds)
        # self._storm.tick(delta_seconds)
        self.weather.cloudiness = 80
        self.weather.precipitation = 75
        self.weather.precipitation_deposits = 80
        self.weather.wind_intensity = 90
        self.weather.fog_density = 15
        self.weather.wetness = 80
        self.weather.sun_azimuth_angle = -20
        self.weather.sun_altitude_angle = 185

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud, state_manager):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.state_manager = state_manager
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity, actor_id in self.history:
            history[frame] += intensity
        return history

    # @staticmethod
    # def _on_collision(weak_self, event):
    #     self = weak_self()
    #     if not self:
    #         return
    #     actor_type = get_actor_display_name(event.other_actor)
    #     self.hud.notification('Collision with %r' % actor_type)
    #     self.state_manager.collision_handler(actor_type)
    #     # print('Collision with %r' % actor_type)
    #     impulse = event.normal_impulse
    #     intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    #     self.history.append((event.frame, intensity))
    #     if len(self.history) > 4000:
    #         self.history.pop(0)

    # Modified versions of above code for resetting based on collision with walkers and vehicles
    def get_collision_ids(self):
        history = collections.defaultdict(str)

        for frame, intensity, actor_id in self.history:
            history[frame] += actor_id
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        self.state_manager.collision_handler(actor_type)
        # print('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity, event.other_actor.type_id))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, actor_id):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            # (carla.Transform(carla.Location(x=-5.5, z=2.5),
            #  carla.Rotation(pitch=8.0)), Attachment.Rigid),


            (carla.Transform(carla.Location(x=0.0, z=50.0), carla.Rotation(pitch=-90)), Attachment.Rigid),
            # (carla.Transform(carla.Location(x=0.3, z=0.8)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=1., z=0.)), Attachment.Rigid),


            # (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            # (carla.Transform(carla.Location(x=-8.0, z=6.0),
            #  carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            # (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)
            ]
        self.transform_index = 1
        self.sensors = [
            # ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {'fov': '110'}]]
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None
        self.actor_id = actor_id

    def toggle_camera(self):
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index]
             [2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' %
                              ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display[self.actor_id].blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]
                    ['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(
                dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


class World(object):
    def __init__(self, carla_world, hud, minimap, state_manager, gamma, rolename, actor_filter, spawn_points):
        self.world = carla_world
        self.actor_ids = rolename
        self.spawn_points = spawn_points
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.minimap = minimap
        self.state_manager = state_manager
        self.player = {}
        self.collision_sensor = {}
        self.lane_invasion_sensor = {}
        self.gnss_sensor = {}
        self.imu_sensor = {}
        self.radar_sensor = {}
        self.camera_manager = {}
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self._gamma = gamma
        self.restart()
        self.world.on_tick(self.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False

        self.path_points = []
        self.count = 0
        #
        self.after_init()

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        spawn_z = 1.5
        # Keep same camera config if the camera manager exists.
        for actor_id, spawn_pt in zip(self.actor_ids, self.spawn_points):
            #cam_index=0
            cam_index = self.camera_manager[actor_id].index if self.camera_manager.get(actor_id) is not None else 0
            #cam_index=1
            cam_pos_index = self.camera_manager[actor_id].transform_index if self.camera_manager.get(actor_id) is not None else 0
            # Get a random blueprint.
            blueprint = random.choice(
                self.world.get_blueprint_library().filter(self._actor_filter))
            blueprint.set_attribute('role_name', actor_id)
            if blueprint.has_attribute('color'):
                color = random.choice(
                    blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(
                    blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'true')
            # set the max speed
            if blueprint.has_attribute('speed'):
                self.player_max_speed = float(
                    blueprint.get_attribute('speed').recommended_values[1])
                self.player_max_speed_fast = float(
                    blueprint.get_attribute('speed').recommended_values[2])
            else:
                print("No recommended values for 'speed' attribute")
            
            print("Spawning Player")
            # Get random sampled spawn points to spawn ego walker
            spawn_point = carla.Transform()
            spawn_point.location.x = spawn_pt[0]
            spawn_point.location.y = spawn_pt[1]
            spawn_point.location.z = spawn_z
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            # spawn_point.rotation.yaw = spawn_pt[5]

            sampled_spawn_points = np.add(np.random.uniform(low=[-0.75,-0.75],high=[0.75,0.75], size=(25,2)), np.array([spawn_pt]))
            if self.player.get(actor_id) is not None:
                self.destroy()
                self.player[actor_id] = self.world.try_spawn_actor(blueprint, spawn_point)
            while self.player.get(actor_id) is None:
                if not self.map.get_spawn_points():
                    print('There are no spawn points available in your map/town.')
                    print('Please add some Vehicle Spawn Point to your UE4 scene.')
                    sys.exit(1)
                
                spawn_x, spawn_y= sampled_spawn_points[np.random.randint(25)]
                spawn_point.location.x = spawn_x #-8
                spawn_point.location.y = spawn_y
                spawn_point.location.z = spawn_z
                spawn_point.rotation.roll = 0.0 #-8
                spawn_point.rotation.pitch = 0.0
                # spawn_point.rotation.yaw = spawn_yaw

                self.player[actor_id] = self.world.try_spawn_actor(blueprint, spawn_point)

            # Set up the sensors.
            self.collision_sensor[actor_id] = CollisionSensor(
                self.player[actor_id], self.hud[actor_id], self.state_manager[actor_id])
            self.gnss_sensor[actor_id] = GnssSensor(self.player[actor_id])
            self.imu_sensor[actor_id] = IMUSensor(self.player[actor_id])
            self.camera_manager[actor_id] = CameraManager(self.player[actor_id], self.hud[actor_id], self._gamma, actor_id)
            self.camera_manager[actor_id].transform_index = cam_pos_index
            self.camera_manager[actor_id].set_sensor(cam_index, notify=False)
            actor_type = get_actor_display_name(self.player[actor_id])
            self.hud[actor_id].notification(actor_type)
            self.minimap[actor_id].tick(self)
    

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        for actor_id in self.actor_ids:
            self.hud[actor_id].notification('Weather: %s' % preset[1])
        self.player[self.actor_ids[0]].get_world().set_weather(preset[0])

    def tick(self, clock):
        # Run during world loop
        for actor_id in self.actor_ids:
            self.hud[actor_id].tick(self, clock)
            self.minimap[actor_id].tick(self)
        # print("Hi")

    def render(self, display):
        for actor_id in self.actor_ids:
            self.camera_manager[actor_id].render(display)
            self.hud[actor_id].render(display)
            self.minimap[actor_id].render(display)
        # pass

    def destroy_sensors(self):
        for actor_id in self.actor_ids:
            self.camera_manager[actor_id].sensor.destroy()
            self.camera_manager[actor_id].sensor = None
            self.camera_manager[actor_id].index = None

    def destroy(self):
        self.before_destroy()
        for actor_id in self.actor_ids:
            sensors = [
                self.camera_manager[actor_id].sensor,
                self.collision_sensor[actor_id].sensor,
                self.gnss_sensor[actor_id].sensor,
                self.imu_sensor[actor_id].sensor]
            for sensor in sensors:
                if sensor is not None:
                    sensor.stop()
                    sensor.destroy()
            if self.player[actor_id] is not None:
                self.player[actor_id].destroy()

    def after_init(self):
        # Run after world initialization
        print('After initialization')
        return

    def before_destroy(self):
        # Run before world destroyed
        print('Before destroy')
        for actor_id in self.actor_ids:
            self.state_manager[actor_id].save_log()
        return

    def on_world_tick(self, timestamp):
        for actor_id in self.actor_ids:
            self.hud[actor_id].on_world_tick(timestamp)
            self.state_manager[actor_id].on_world_tick(timestamp)


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world, actor_id):
        self.actor_id = actor_id
        if isinstance(world.player[self.actor_id], carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player[self.actor_id].set_light_state(self._lights)
        elif isinstance(world.player[self.actor_id], carla.Walker):
            self._control = carla.WalkerControl()
            self._rotation = world.player[self.actor_id].get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud[self.actor_id].notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    world.hud[self.actor_id].toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud[self.actor_id].help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager[self.actor_id].toggle_camera()
                    world.minimap[self.actor_id].tick(world)
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player[self.actor_id].disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud[self.actor_id].notification(
                            "Disabled Constant Velocity Mode")
                    else:
                        world.player[self.actor_id].enable_constant_velocity(
                            carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud[self.actor_id].notification(
                            "Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager[self.actor_id].set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager[self.actor_id].toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud[self.actor_id].notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud[self.actor_id].notification("Recorder is ON")

        # if not self._autopilot_enabled:
        if isinstance(self._control, carla.WalkerControl):
            self._parse_walker_keys(
                pygame.key.get_pressed(), clock.get_time(), world)
        else:
            raise ('Invalid Controller')
        world.player[self.actor_id].apply_control(self._control)

    def _parse_walker_keys(self, keys, milliseconds, world):
        print("Enter Keyboard input")
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            # self._rotation.yaw -= 0.04 * milliseconds
            self._rotation.yaw -= 0.005 * milliseconds

        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            # self._rotation.yaw += 0.04 * milliseconds
            self._rotation.yaw += 0.005 * milliseconds


        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods(
            ) & KMOD_SHIFT else world.player_max_speed


        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height, actor_id):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.actor_id = actor_id

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player[self.actor_id].get_transform()
        v = world.player[self.actor_id].get_velocity()
        c = world.player[self.actor_id].get_control()
        compass = world.imu_sensor[self.actor_id].compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor[self.actor_id].get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            # 'Server:  % 16.0f FPS' % self.server_fps,
            # 'Client:  % 16.0f FPS' % clock.get_fps(),
            # '',
            'Agent: % 20s' % get_actor_display_name(
                world.player[self.actor_id], truncate=20),
            'Map:     % 20s' % world.map.name,
            # 'Simulation time: % 12s' % datetime.timedelta(
            #     seconds=int(self.simulation_time)),
            'Experiment time: %12s' % datetime.timedelta(seconds=int(
                self.simulation_time - world.state_manager[self.actor_id].init_t)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor[self.actor_id].accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor[self.actor_id].gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' %
                            (world.gnss_sensor[self.actor_id].lat, world.gnss_sensor[self.actor_id].lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            def distance(l): return math.sqrt((l.x - t.location.x)**2 +
                                              (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x)
                        for x in vehicles if x.id != world.player[self.actor_id].id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        # record

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display[self.actor_id].blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display[self.actor_id], (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display[self.actor_id], (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display[self.actor_id], (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display[self.actor_id], (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display[self.actor_id].blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display[self.actor_id])
        self.help.render(display[self.actor_id])


# ==============================================================================
# -- MiniMap ----------------------------------------------------------------
# ==============================================================================

class MiniMap(object):
    def __init__(self, path_dir, actor_id):
        self.planner = global_planner.Planner(path_dir)
        self.actor_id = actor_id
        return

    def tick(self, world):
        # player info
        self.player_pos = world.player[self.actor_id].get_transform().location
        # Camera info
        camera_width = float(
            world.camera_manager[self.actor_id].sensor.attributes['image_size_x'])
        camera_height = float(
            world.camera_manager[self.actor_id].sensor.attributes['image_size_y'])
        camera_fov = float(world.camera_manager[self.actor_id].sensor.attributes['fov'])
        focal = camera_width/(2*np.tan(camera_fov * np.pi / 360))

        world_points = self.planner.path_pts[:, :2]
        path_pts = self._world_to_camera(
            world.camera_manager[self.actor_id].sensor, world_points, 1)
        # filter out points with negative x coord (behind camera)
        path_pts = path_pts[:, path_pts[0, :] > 0]
        # apply transformation
        points = self._camera_to_image(camera_width, camera_fov, path_pts)
        # # convert to pygame coord
        self.points = self._image_to_screen(
            camera_width, camera_height, points)

        self.rects = []
        self.ego_rect = None
        marker_size = 0.1
        ego_size = 4
        delta = marker_size*focal / path_pts[0, :]
        # for idx, pt in enumerate(self.points):
        #     self.rects += [pygame.Rect(pt[0]-delta[idx]/2,
        #                                pt[1]-delta[idx]/2, delta[idx], delta[idx])]

        # if camera is topdown camera
        self._points_vis = []
        if world.camera_manager[self.actor_id].transform_index == 0:

            self._points_vis = self.points

            for idx, pt in enumerate(self.points):
                self.rects += [pygame.Rect(pt[0]-delta[idx]/2,
                                           pt[1]-delta[idx]/2, delta[idx], delta[idx])]

            ego_world_pos = np.array([[self.player_pos.x, self.player_pos.y]])
            #print("Inside rh",ego_world_pos)
            ego_pt = self._world_to_camera(
                world.camera_manager[self.actor_id].sensor, ego_world_pos, 1)
            ego_pt = self._camera_to_image(camera_width, camera_fov, ego_pt)
            ego_pt = self._image_to_screen(camera_width, camera_height, ego_pt)
            self.ego_rect = pygame.Rect(
                ego_pt[0][0]-ego_size/2, ego_pt[0][1]-ego_size/2, ego_size, ego_size)

        # temp: visualizing local goal
        self.goal_rect = None

        local_goal = self.planner.get_next_goal(
            pos=[self.player_pos.x, self.player_pos.y], preview_s=5)
        local_goal = np.array(local_goal)[None, :]
        #print("Inside rh",local_goal)
        local_goal = self._world_to_camera(
            world.camera_manager[self.actor_id].sensor, local_goal, 1)
        local_goal = self._camera_to_image(
            camera_width, camera_fov, local_goal)
        local_goal = self._image_to_screen(
            camera_width, camera_height, local_goal)
        #print("Local Goal inside rh", local_goal)
        #print("Ego Point inside rh",ego_pt)
        self.goal_rect = pygame.Rect(
            local_goal[0][0]-ego_size/2, local_goal[0][1]-ego_size/2, ego_size, ego_size)
        return

    def _world_to_camera(self, camera, coord, z_offset):
        # input: Nx3 matrix, [[x1, y1, z1], [x2, y2, z1], ...]
        # output: 3xN matrix [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
        camera_pos = camera.get_transform()
        camera_world_pos = np.array(camera_pos.get_matrix())
        world_camera_pos = np.linalg.inv(camera_world_pos)

        coord = coord.swapaxes(0, 1)
        coord = np.vstack([coord, np.zeros(coord.shape[1])+z_offset,
                          np.ones(coord.shape[1])])  # adding height = 1
        return world_camera_pos @ coord

    def _camera_to_image(self, width, fov, coord):
        # input: 3xN matrix [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
        # output: 3xN matrix [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
        focal = width/(2*np.tan(fov * np.pi / 360))
        reg_coord = coord[:3, :] / coord[0, :]
        mat = np.ones((3, 3))
        mat[1, 1] = mat[2, 2] = focal
        return mat @ reg_coord

    def _image_to_screen(self, width, height, coord):
        # input: 3xN matrix [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
        # output: Nx2 matrix [[x1, y1], [x2, y2], ...]
        coord = coord.swapaxes(0, 1)
        pts = np.zeros((coord.shape[0], 2))
        pts[:, 0] = width/2 + coord[:, 1]
        pts[:, 1] = height/2 - coord[:, 2]
        return pts

    def render(self, display):
        # if len(self.points) > 1:
        if len(self._points_vis) > 1:
            pygame.draw.lines(display[self.actor_id], (255, 0, 0), False, self._points_vis)

            pygame.draw.rect(display[self.actor_id], (0, 0, 255), self.goal_rect)

        for rect in self.rects:
            pygame.draw.rect(display[self.actor_id], (255, 0, 0), rect)

        if self.ego_rect:
            pygame.draw.rect(display[self.actor_id], (0, 255, 0), self.ego_rect)

        # if self.goal_rect:
        #     pygame.draw.rect(display, (0, 0, 255), self.goal_rect)
        return

# ==============================================================================
# -- StateManager --------------------------------------------------------------
# ==============================================================================

# handle collision
# handle timeout
# handle goal reached
# log info


class StateManager(object):
    def __init__(self, minimap, log_dir):
        self.log = {'config': CONF, 'event': []}
        self.log_dir = Path(log_dir)
        self.end_state = False
        self.start_state = False
        self.time_limit_seconds = 100000000000
        self.simulation_time = 0
        self.init_t = 0
        self.last_collision_t = 0
        self.last_collision_type = None
        self.minimap = minimap
        return

    def on_world_tick(self, timestamp):
        self.simulation_time = timestamp.elapsed_seconds
        self.timeout_handler(self.simulation_time)
        if self.start_state:
            self.success_handler(self.simulation_time)
        return

    def start_handler(self, timestamp):
        self.start_state = True
        self.end_state = False
        self.init_t = self.simulation_time
        self.log['path_dir'] = self.minimap.planner.path_file
        self.log['event'].append(self._event(0,  'Start experiment'))
        print(self.log)
        return

    def success_handler(self, timestamp):
        radius = 0.2 # reached goal if distance < radius
        player_pos = np.array([self.minimap.player_pos.x,
                              self.minimap.player_pos.y])
        goal = self.minimap.planner.path_pts[-1, 0:2]
        d = np.sqrt(np.sum((player_pos - goal)**2))
        if d < radius:
            # success
            delta_t = timestamp - self.init_t
            self.log['event'].append(self._event(delta_t, "Goal reached"))
            self._end(success=True, info=self._event(delta_t, "Goal reached"))
        return

    def collision_handler(self, actor_type):
        # TODO: handle walker and vehicle colliison
        freq_thresh = 1.0
        delta_t = self.simulation_time - self.init_t
        if actor_type == 'Sidewalk':
            return

        if delta_t - self.last_collision_t > freq_thresh or actor_type != self.last_collision_type:
            self.log['event'].append(self._event(
                delta_t, 'Collide with %s' % actor_type))
            self.last_collision_t = delta_t
            self.last_collision_type = actor_type
            print(delta_t, actor_type)
            if actor_type == 'Vehicle':
                self._end(success=False, info=self._event(
                    delta_t, 'Collide with %s' % actor_type))
        return

    def timeout_handler(self, timestamp):
        if self.start_state:
            delta_t = timestamp - self.init_t
            if delta_t > self.time_limit_seconds:
                self.log['event'].append(self._event(delta_t, "Time out"))
                self._end(success=False, info=self._event(delta_t, "Time out"))
                print(self.log)
        return

    def _end(self, success, info):
        self.log['result'] = {'success': success, 'info': info}
        self.start_state = False
        self.end_state = True

    @staticmethod
    def _event(timestamp, event_text):
        return {'timestamp': timestamp, 'info': event_text}

    def save_log(self):
        if not os.path.exists(str(self.log_dir)):
            self.log_dir.mkdir(parents=True, exist_ok=True)
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        save_path = self.log_dir/(self.minimap.actor_id + "_" + time + '_log.json')
        with open(save_path, 'w') as f:
            json.dump(self.log, f)
            print('Log saved to: ',save_path)
        return

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)




# Camera spawn methods

def spawn_camera(world, vehicle, _type, w, h, fov, x, y, z, pitch, yaw):
    if _type == 'rgb':
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    elif _type == 'instance_segmentation':
        camera_bp = world.get_blueprint_library().find(
            'sensor.camera.instance_segmentation')
    elif _type == 'semantic_segmentation':
        camera_bp = world.get_blueprint_library().find(
            'sensor.camera.semantic_segmentation')
    elif _type == 'depth':
        camera_bp = world.get_blueprint_library().find('sensor.camera.depth')

    camera_bp.set_attribute('image_size_x', str(w))
    camera_bp.set_attribute('image_size_y', str(h))
    camera_bp.set_attribute('fov', str(fov))
    _camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=x, y=y, z=z),
                        carla.Rotation(pitch=pitch, yaw=yaw)),
        attach_to=vehicle)

    view_width = w
    view_height = h
    view_fov = fov
    if _type == 'rgb':
        calibration = np.identity(3)
        calibration[0, 2] = view_width / 2.0
        calibration[1, 2] = view_height / 2.0
        calibration[0, 0] = calibration[1, 1] = view_width / \
            (2.0 * np.tan(view_fov * np.pi / 360.0))
        _camera.calibration = calibration

    return _camera


# Spawn multiple cameras on actor

def spawn_cameras(world, walker, width, height, fov):
    cameras = {}
    actor = walker
    # eye-level camera
    x_forward_eyelevel = 1  # 0.4
    cam_height_eyelevel = 0.0 #0.8  # 0.5

    fov = fov
    w = width
    h = height

    cameras['eyelevel_rgb'] = spawn_camera(world, actor, 'rgb', w, h, fov,
                                           x_forward_eyelevel, 0.0, cam_height_eyelevel, 0.0, 0.0)

    cameras['eyelevel_ins'] = spawn_camera(world, actor, 'instance_segmentation', w, h, fov,
                                           x_forward_eyelevel, 0.0, cam_height_eyelevel, 0.0, 0.0)


    cameras['eyelevel_dep'] = spawn_camera(world, actor, 'depth', w, h, fov,
                                            x_forward_eyelevel, 0.0, cam_height_eyelevel, 0.0, 0.0)

    return cameras





class CarlaMultiActorEnvPZ(AECEnv, EzPickle):
    """
    Wrapper for CARLA MultiActor environment implmenting AECEnv interface from PettingZoo
    """
    def __init__(self, **kwargs):
        """
        Initialize the environment.
        """
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.env = CarlaMultiActorEnv(**self._kwargs)

        self.metadata = {**self.env.metadata, "is_parallelizable": True}

    @property
    def observation_spaces(self):
        return self.env.observation_spaces
    
    @property
    def action_spaces(self):
        return self.env.action_spaces

    def observation_space(self, actor_id: str):
        return self.env.observation_space(actor_id)

    def action_space(self, actor_id: str):
        return self.env.action_space(actor_id)
    

    @property
    def agents(self):
        return self.env.actor_ids

    @property
    def possible_agents(self):
        return self.env.actor_ids
    
    @property
    def num_agents(self):
        return self.env.num_walkers
    
    @property
    def terminations(self):
        return self.env._terminations.copy()

    @property
    def truncations(self):
        return self.env._truncations.copy()
    
    def reset(self, seed=None, options=None):
        self.prev_obs = self.env.reset()
        self._actions: ActionDict = {agent: None for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def observe(self, agent: str):
        obs = self.env.prev_obs[agent]
        return obs

    
    def step(self, action):
        if not self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._actions[self.agent_selection] = action
            if self._agent_selector.is_last():
                _, rewards, _, _, infos = self.env.step(self._actions)

                self.rewards = rewards.copy()
                self.infos = deepcopy(infos)
                self._accumulate_rewards()

                if len(self.env.actor_ids):
                    self._agent_selector = agent_selector(self.env.actor_ids)
                    self.agent_selection = self._agent_selector.reset()
                
                self._deads_step_first()
            else:
                if self._agent_selector.is_first():
                    self._clear_rewards()
                
                self.agent_selection = self._agent_selector.next()

    
    def last(self, observe: bool = True):
        agent = self.agent_selection
        obs = self.observe(agent) if observe else None,

        return (obs, self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent])



    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()

    
    def __str__(self):
        return str(self.env)