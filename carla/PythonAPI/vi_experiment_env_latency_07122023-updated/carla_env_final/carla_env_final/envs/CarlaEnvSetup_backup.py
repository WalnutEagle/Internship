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

# logging.set_verbosity(logging.DEBUG)

CONF = {"tm_seed": 2021,  # seed for traffic manager
        "car_lights_on": False,
        "percent_walking": 0,  # how many perdestrians will walk
        "percent_crossing": 0,  # how many pedestrians will cross road
        "percent_disabled": 0,
        "max_render_depth_in_meters": 100,
        "min_visible_vertices_for_render": 4,
        "num_vehicles": 200,
        "num_walkers": 30 # Changed from 50 to 0
        }

def setup1():
    port = 2000
    server_timestop=10.0
    #log_level = logging.DEBUG if args.debug else logging.INFO
    #logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # logging.info('Starting server %s: %s'. args.host, args.port)
    env = os.environ.copy()
    # env["SDL_VIDEODRIVER"] = "offscreen"
    # env["SDL_HINT_CUDA_DEVICE"] = "0"
    env["CARLA_ROOT"]="/data2/kathakoli/carla/Unreal/CarlaUE4/Saved/StagedBuilds/LinuxNoEditor/"
    #env["CUDA_VISIBLE_DEVICES"]= "1"
    #logging.debug("Inits a CARLA server at port={}".format(port))
    #CUDA_VISIBLE_DEVICES=3 ' + 
    server = subprocess.Popen(str(os.path.join(env["CARLA_ROOT"], "CarlaUE4.sh")), stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, shell=True)
    atexit.register(os.killpg, server.pid, signal.SIGKILL)
    time.sleep(server_timestop)

    print(__doc__)
    return server


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
    display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

    tag_set = list(fme.Tags.Hash2.values())
    path_finder = ig.PathFinder()
    world = None
    vehicles_list = []
    all_id_dict = {}
    port=args.port
    # Connect client.
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    #logging.info('Starting server %s: %s'. args.host, args.port)
    env = os.environ.copy()
    # env["SDL_VIDEODRIVER"] = "offscreen"
    # env["SDL_HINT_CUDA_DEVICE"] = "0"
    env["CARLA_ROOT"]="/data2/kathakoli/carla/Unreal/CarlaUE4/Saved/StagedBuilds/LinuxNoEditor/"
    #env["CUDA_VISIBLE_DEVICES"]= "1"
    logging.debug("Inits a CARLA server at port={}".format(port))
    #CUDA_VISIBLE_DEVICES=3 ' + 
    server = subprocess.Popen(str(os.path.join(env["CARLA_ROOT"], "CarlaUE4.sh -RenderOffScreen")), stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, shell=True)
    atexit.register(os.killpg, server.pid, signal.SIGKILL)
    time.sleep(server_timestop)

    print(__doc__)
    logging.debug("Connects a CARLA client at port={}".format(port))
    
    # Setting up CARLA client
    client = carla.Client(args.host, args.port)
    client.set_timeout(client_timeout)
    hud = HUD(args.width, args.height)
    minimap = MiniMap(args.path)
    state_manager = StateManager(minimap, args.log_dir)
    # print(args.world)
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

    # vehicles_list = util.spawn_vehicles(client, world.world, CONF['num_vehicles'], True, traffic_manager)
    # pedestrian_seed = 42
    # world.world.set_pedestrians_seed(pedestrian_seed)  
    s=[]
    if not util.spawn_walkers(client, world.world, 30, 0, 0.9, 0.2, all_id_dict):
        print("spawn_walkers failed")
        return
    
    # spawn dense
    # dense_N = 100
    # dense_percent_disabled = 0.0
    player_location = world.player.get_transform()
    loc_center = [player_location.location.x, player_location.location.y, player_location.location.z]

    controller = KeyboardControl(world)

    # print("loc_center: ", loc_center)
    val1=util.spawn_walkers_dense(client, world.world, 30, loc_center,0, 0.9, 0.2, all_id_dict)
    if not val1:
        print("spawn_walkers_dense failed")
        return
    # cameras = spawn_cameras(world.world, world.player, world.destination_vehicle, args.width, args.height, 90)
    # state_manager.start_handler(world)

    all_actors = world.world.get_actors()
    all_vehicles = []
    all_peds = []
    for _a in all_actors:
        if 'vehicle' in _a.type_id:
            # print(_a.type_id)
            all_vehicles.append(_a)

        if  _a.type_id.startswith('walker'):
            # print(_a.type_id)
            all_peds.append(_a)
    
    return client, world,server, minimap, state_manager, traffic_manager, controller, display, all_id_dict, vehicles_list,s
    




