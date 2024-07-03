#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import copy
import carla_env_final
import math
sys.path.append('data2/kathakoli/carla/PythonAPI/vi_experiment_env_latency_07122023/carla_env_final/carla_env_final/envs/')
from carla_env_final.envs.CarlaEnvSetup import setup,docker_setup
from CarlaSyncMode import CarlaSyncMode

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
from local_model import ClassificationNetwork

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
import gymnasium as gym
from gymnasium import spaces
import time

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
from queue import Queue
import json
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

import global_planner
from CarlaSyncMode import CarlaSyncMode
import freemap_extraction as fme
from freemap import Freemap
import instruction_generation as ig
import util
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from stable_baselines3 import PPO
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

#from run_experiment_xworld_rh import HUD, World, MiniMap, StateManager, spawn_cameras, KeyboardControl

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


pairs = [['crl_hips__C', 'crl_spine__C'], ['crl_hips__C', 'crl_thigh__L'], ['crl_hips__C', 'crl_thigh__R'],
        ['crl_spine__C', 'crl_spine01__C'], ['crl_spine01__C', 'crl_shoulder__L'], ['crl_spine01__C', 'crl_neck__C'], ['crl_spine01__C', 'crl_shoulder__R'],
        
        ['crl_shoulder__L', 'crl_arm__L'], ['crl_arm__L', 'crl_foreArm__L'], ['crl_foreArm__L', 'crl_hand__L'], 
        ['crl_hand__L', 'crl_handThumb__L'], ['crl_hand__L', 'crl_handIndex__L'], ['crl_hand__L', 'crl_handMiddle__L'], ['crl_hand__L', 'crl_handRing__L'], ['crl_hand__L', 'crl_handPinky__L'],
        ['crl_handThumb__L', 'crl_handThumb01__L'], ['crl_handThumb01__L', 'crl_handThumb02__L'], ['crl_handThumb02__L', 'crl_handThumbEnd__L'],
        ['crl_handIndex__L', 'crl_handIndex01__L'], ['crl_handIndex01__L', 'crl_handIndex02__L'], ['crl_handIndex02__L', 'crl_handIndexEnd__L'],
        ['crl_handMiddle__L', 'crl_handMiddle01__L'], ['crl_handMiddle01__L', 'crl_handMiddle02__L'], ['crl_handMiddle02__L', 'crl_handMiddleEnd__L'],
        ['crl_handRing__L', 'crl_handRing01__L'], ['crl_handRing01__L', 'crl_handRing02__L'], ['crl_handRing02__L', 'crl_handRingEnd__L'],
        ['crl_handPinky__L', 'crl_handPinky01__L'], ['crl_handPinky01__L', 'crl_handPinky02__L'], ['crl_handPinky02__L', 'crl_handPinkyEnd__L'],

        ['crl_neck__C', 'crl_Head__C'], ['crl_Head__C', 'crl_eye__L'], ['crl_Head__C', 'crl_eye__R'],

        ['crl_shoulder__R', 'crl_arm__R'], ['crl_arm__R', 'crl_foreArm__R'], ['crl_foreArm__R', 'crl_hand__R'], 
        ['crl_hand__R', 'crl_handThumb__R'], ['crl_hand__R', 'crl_handIndex__R'], ['crl_hand__R', 'crl_handMiddle__R'], ['crl_hand__R', 'crl_handRing__R'], ['crl_hand__R', 'crl_handPinky__R'],
        ['crl_handThumb__R', 'crl_handThumb01__R'], ['crl_handThumb01__R', 'crl_handThumb02__R'], ['crl_handThumb02__R', 'crl_handThumbEnd__R'],
        ['crl_handIndex__R', 'crl_handIndex01__R'], ['crl_handIndex01__R', 'crl_handIndex02__R'], ['crl_handIndex02__R', 'crl_handIndexEnd__R'],
        ['crl_handMiddle__R', 'crl_handMiddle01__R'], ['crl_handMiddle01__R', 'crl_handMiddle02__R'], ['crl_handMiddle02__R', 'crl_handMiddleEnd__R'],
        ['crl_handRing__R', 'crl_handRing01__R'], ['crl_handRing01__R', 'crl_handRing02__R'], ['crl_handRing02__R', 'crl_handRingEnd__R'],
        ['crl_handPinky__R', 'crl_handPinky01__R'], ['crl_handPinky01__R', 'crl_handPinky02__R'], ['crl_handPinky02__R', 'crl_handPinkyEnd__R'],

        ['crl_thigh__L', 'crl_leg__L'], ['crl_leg__L', 'crl_foot__L'], ['crl_foot__L', 'crl_toe__L'], ['crl_toe__L', 'crl_toeEnd__L'],
        ['crl_thigh__R', 'crl_leg__R'], ['crl_leg__R', 'crl_foot__R'], ['crl_foot__R', 'crl_toe__R'], ['crl_toe__R', 'crl_toeEnd__R'],

]


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


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

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, minimap, state_manager, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
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
        self.player = None
        self.destination_vehicle = None
        self.path_dir = args.path
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.start_id = args.start_id
        self.destination_id = args.destination_id
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
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
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
        # Spawn the player.]
        print("Spawning Player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            # spawn_point = random.choice(
            #     spawn_points) if spawn_points else carla.Transform()\
            spawn_point = spawn_points[self.start_id]
            print('location0: ', spawn_point.location.x, spawn_point.location.y, spawn_point.location.z)

            # spawn_point.location.x = 13 #32.6081
            # spawn_point.location.y = 59 #39.0366

            # -7.966956615447998 66.28325653076172
            # spawn_point.location.x = -15 #-8
            # spawn_point.location.y = 59


            # spawn_point.location.x = -30 #-8
            # spawn_point.location.y = 57

            spawn_point.location.x = -25 #-8
            spawn_point.location.y = 58

            spawn_point.rotation.yaw = -180

            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        veh_blueprint = random.choice(
            # self.world.get_blueprint_library().filter('cybertruck'))
            self.world.get_blueprint_library().filter('model3'))

        veh_blueprint.set_attribute('color', '128,128,128')

        while self.destination_vehicle is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[self.destination_id]


            # spawn_point.location.x = -41.668877
            spawn_point.location.x = -38.668877
            spawn_point.location.y = 48.905540
            spawn_point.location.z = 0.6
            spawn_point.rotation.yaw = -90.161217


            self.destination_vehicle = self.world.try_spawn_actor(
                veh_blueprint, spawn_point)
            print("Destination Vehicle spawned at: ", spawn_point)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(
            self.player, self.hud, self.state_manager)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.minimap.tick(self)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        # Run during world loop
        self.hud.tick(self, clock)
        self.minimap.tick(self)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)
        self.minimap.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        self.before_destroy()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        if self.destination_vehicle is not None:
            self.destination_vehicle.destroy()

    def after_init(self):
        # Run after world initialization
        print('After initialization')
        return

    def before_destroy(self):
        # Run before world destroyed
        print('Before destroy')
        self.state_manager.save_log()
        return

    def on_world_tick(self, timestamp):
        self.hud.on_world_tick(timestamp)
        self.state_manager.on_world_tick(timestamp)

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world):
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

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
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                    world.minimap.tick(world)
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification(
                            "Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(
                            carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification(
                            "Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")

        # if not self._autopilot_enabled:
        if isinstance(self._control, carla.WalkerControl):
            self._parse_walker_keys(
                pygame.key.get_pressed(), clock.get_time(), world)
        else:
            raise ('Invalid Controller')
        world.player.apply_control(self._control)

    def _parse_walker_keys(self, keys, milliseconds, world):
        print("Enter keyboard input")
        self._control.speed = 0.0
        #pygame.event.clear()
        #event = pygame.event.wait()
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
    def __init__(self, width, height):
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

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            # 'Server:  % 16.0f FPS' % self.server_fps,
            # 'Client:  % 16.0f FPS' % clock.get_fps(),
            # '',
            'Agent: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            # 'Simulation time: % 12s' % datetime.timedelta(
            #     seconds=int(self.simulation_time)),
            'Experiment time: %12s' % datetime.timedelta(seconds=int(
                self.simulation_time - world.state_manager.init_t)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' %
                            (world.gnss_sensor.lat, world.gnss_sensor.lon)),
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
                        for x in vehicles if x.id != world.player.id]
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
            display.blit(info_surface, (0, 0))
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
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- MiniMap ----------------------------------------------------------------
# ==============================================================================

class MiniMap(object):
    def __init__(self, path_dir):
        self.planner = global_planner.Planner(path_dir)
        return

    def tick(self, world):
        # player info
        self.player_pos = world.player.get_transform().location
        # Camera info
        camera_width = float(
            world.camera_manager.sensor.attributes['image_size_x'])
        camera_height = float(
            world.camera_manager.sensor.attributes['image_size_y'])
        camera_fov = float(world.camera_manager.sensor.attributes['fov'])
        focal = camera_width/(2*np.tan(camera_fov * np.pi / 360))

        world_points = self.planner.path_pts[:, :2]
        path_pts = self._world_to_camera(
            world.camera_manager.sensor, world_points, 1)
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
        if world.camera_manager.transform_index == 0:

            self._points_vis = self.points

            for idx, pt in enumerate(self.points):
                self.rects += [pygame.Rect(pt[0]-delta[idx]/2,
                                           pt[1]-delta[idx]/2, delta[idx], delta[idx])]

            ego_world_pos = np.array([[self.player_pos.x, self.player_pos.y]])
            ego_pt = self._world_to_camera(
                world.camera_manager.sensor, ego_world_pos, 1)
            ego_pt = self._camera_to_image(camera_width, camera_fov, ego_pt)
            ego_pt = self._image_to_screen(camera_width, camera_height, ego_pt)
            self.ego_rect = pygame.Rect(
                ego_pt[0][0]-ego_size/2, ego_pt[0][1]-ego_size/2, ego_size, ego_size)

        # temp: visualizing local goal
        self.goal_rect = None

        local_goal = self.planner.get_next_goal(
            pos=[self.player_pos.x, self.player_pos.y], preview_s=5)
        local_goal = np.array(local_goal)[None, :]
        local_goal = self._world_to_camera(
            world.camera_manager.sensor, local_goal, 1)
        local_goal = self._camera_to_image(
            camera_width, camera_fov, local_goal)
        local_goal = self._image_to_screen(
            camera_width, camera_height, local_goal)
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
            pygame.draw.lines(display, (255, 0, 0), False, self._points_vis)

            pygame.draw.rect(display, (0, 0, 255), self.goal_rect)

        for rect in self.rects:
            pygame.draw.rect(display, (255, 0, 0), rect)

        if self.ego_rect:
            pygame.draw.rect(display, (0, 255, 0), self.ego_rect)

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
        # print(self.log)
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
        save_path = self.log_dir/(time + '_log.json')
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
        for frame, intensity in self.history:
            history[frame] += intensity
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
        self.history.append((event.frame, intensity))
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
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = True
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
            display.blit(self.surface, (0, 0))

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


# ==============================================================================
# -- Functions For experiment --------------------------------------------------
# ==============================================================================

# def before_loop(world):
#     spawn_points = world.map.get_spawn_points()
#     spawn_point_np = np.zeros((len(spawn_points), 3))
#     for idx, spawn_point in enumerate(spawn_points):
#         spawn_point_np[idx] = [spawn_point.location.x,
#                                spawn_point.location.y, spawn_point.location.z]
#     np.save('./Town05_spawn_points.npy', spawn_point_np)
#     return


def spawn_fixed_camera(world, _type, w, h, fov, x, y, z, pitch, yaw):
    if _type == 'rgb':
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
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
                        carla.Rotation(pitch=pitch, yaw=yaw)))

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


# spawn_point.location.x = -8
# spawn_point.location.y = 59


def spawn_cameras(world, walker, destination_vehicle, width, height, fov):
    cameras = {}
    actor = walker
    # eye-level camera
    x_forward_eyelevel = 1  # 0.4
    cam_height_eyelevel = 0.0 #0.8  # 0.5

    fov = 120
    w = 1024# 1280
    h = 576 # 720

    cameras['eyelevel_rgb'] = spawn_camera(world, actor, 'rgb', w, h, fov,
                                           x_forward_eyelevel, 0.0, cam_height_eyelevel, 0.0, 0.0)

    cameras['eyelevel_ins'] = spawn_camera(world, actor, 'instance_segmentation', w, h, fov,
                                           x_forward_eyelevel, 0.0, cam_height_eyelevel, 0.0, 0.0)


    cameras['eyelevel_dep'] = spawn_camera(world, actor, 'depth', w, h, fov,
                                            x_forward_eyelevel, 0.0, cam_height_eyelevel, 0.0, 0.0)

    return cameras


def generate_instruction(global_planner, local_planner, x, y, yaw, semantics_img, preview_s=5):
    start = (750, 750)
    mat = fme.get_rotation_mat_z(yaw)
    goal_x, goal_y = global_planner.get_next_goal([x, y], preview_s)
    loc_vec = mat@[goal_x - x, goal_y - y, 0]
    next_goal = (int(loc_vec[1] / 0.02 + 750), int(loc_vec[0]/0.02 + 750))
    # print('location: ', x, y, yaw)
    # print('next_goal: ', next_goal, (goal_x, goal_y), np.linalg.norm([goal_x - x, goal_y - y]), np.linalg.norm([(next_goal[0] - 750)*0.02, (next_goal[1] - 750)*0.02]))
    set_timelimit(2)
    result = None
    try:
        result = local_planner.astar(start, next_goal)
    except Exception:
        pass
    set_timelimit(0)
    # print('before', start, next_goal)
    if result != None:
        path = np.array(list(result))
        ins = ig.InstructionGeneration(
            (0, 0, 0), path, semantics_img, m_per_pixel=0.02)
        instruction, instruction_cluster, poc = ins.get_instructions()
        description, description_cluster, des_poc = ins.get_description(poc)
        print("Instructions: ", instruction)
        print("Description: ", description)
    else:
        generate_instruction(global_planner, local_planner,
                             x, y, yaw, semantics_img, preview_s=1)


def signal_handler(signum, frame):
    raise Exception("Timed out!")


def set_timelimit(seconds):
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    return

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

# # ==============================================================================
# # -- Carla Environment----------------------------------------------------------------
# # ==============================================================================

class CarlaPPOEmbEnv(gym.Env):

    metadata = {'render_modes': ['human']}
    def __init__(self, args,vehicles_list,all_id_dict):
        #self.server=setup1()
        self.client, self.world, self.server,self.minimap, self.state_manager, self.traffic_manager,self.controller,self.display, self.sparse_id_dict, self.dense_id_dict, self.vehicles_list,self.s=docker_setup()        #print(self.s)
        self.total_reward = 0
        self.l2_distance=0
        self.distance_covered=0
        self.im_width = args['width']
        self.im_height = args['height']
        self.repeat_action = args['repeat_action']
        
        self.steps_per_episode =int(args['steps_per_episode'])

        self.clock = pygame.time.Clock()
        self.count=0
        #self.collision_count=0
        self.task_metrics=[]
        self.count_epi=0
        self.action_list=np.array([[0.0]*737 for _ in range(4)])
        self.sampled_latency=0
        self.q=0
        self.ct=0
        self.prev_angle=0.0
        #self.global_model = torch.load("/data2/kathakoli/carla/PythonAPI/vi_experiment_env_latency_07122023/model_10_11_RegNet.pth")
        #self.model=torch.load("/data2/kathakoli/carla/PythonAPI/vi_experiment_env_latency_07122023/local_model_10_11_mobilenet.pth")
        self.l=torch.load("/data/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/Local_model_regnet_14_11.pth")
        self.l=self.l.to(torch.device('cuda'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        self.transform2= transforms.Resize((96,96),antialias=True)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #self.a=torch.Tensor([0.0]*2560)
        self.waypt=np.array([[0.0,0.0]])
        self.pedestrian_detected=0.0
        # l=os.listdir('collection_data')
        # l=sorted(l)
        # if len(l)==0:
        #     new_dir='wk_1'
        # else:
        #     find=l[-1].split('_')
        #     new_dir=find[0]+'_'+str(int(find[-1])+1)
        #     print(l[-1])
        # print(new_dir)
        
        # if not os.path.exists(str('collection_data/')+str(new_dir)):
        #     os.makedirs(str('collection_data/')+str(new_dir))
        # self.wk_path =str('collection_data/')+str(new_dir)
        

        
        #self.last_action=[0,0,0]
        # self._settings = self.world.world.get_settings()
        # self.delta_seconds = 1.0 / 20
        # self.world.world.apply_settings(carla.WorldSettings(
        #     no_rendering_mode=False,
        #     synchronous_mode=True,
        #     fixed_delta_seconds=self.delta_seconds))
        #self.route_completion=0
        

    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        # pedestrian_detected = spaces.Discrete(2)
        # collision_detected = spaces.Discrete(2)
        # distance_covered=spaces.Box(low=0.0,high=np.inf, shape=(1,), dtype=np.float64)
        #return spaces.Box(low=np.array([0.0, 0.0, 0.0,0.0]), high=np.array([1.0, 1.0, np.inf,np.inf]), dtype=np.float64)
        # low_value=[0.0]*2561 for _ in range(4)
        # high_value=[1.0]*2560
        return spaces.Box(low=np.array([[0.0]*737 for _ in range(4)]), high=np.array([[1.0]*737 for _ in range(4)]), dtype=np.float64)
        #0.0,, np.inf 
        # return spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float64)
        # return spaces.Dict({"pedestrian_detected": pedestrian_detected, "collision_detected": collision_detected, "distance_covered": distance_covered})

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        #return spaces.Box(low=np.array([[-3.14, 0.0]]), high=np.array([[3.14, 2.0]]),shape=(1,2), dtype=np.float64)
        #return spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float64)
        return spaces.Discrete(2)

    # def seed(self, seed):
    #     if not seed:
    #         seed = 2021
    #     random.seed(seed)
    #     self._np_random = np.random.RandomState(seed) 
    #     return seed
    def local_policy(self, image, location, pedestrian_detected):
        # imaget=self.transform(image)
        # image_1= self.transform2(imaget)#resize the image
        # normalized_image = self.normalize(image_1.float()/ 255.0)
        # model = torch.load('model_local_only_1.pth')#evaluated the model
        # output = model.forward(normalized_image.unsqueeze(0))#action class output
        # #print(type(output))
        # output_numpy=output.to('cpu').detach().numpy()
        # self.action_list = np.vstack((self.action_list, np.array([output_numpy[0][0]* (math.pi / 180), output_numpy[0][1],0.0])))
        # print("Local Angle",output_numpy[0][0]* (math.pi / 180))
        # return self.action_list[-4:]
        # if pedestrian_detected==0:
        #     self.speed_new = 1.5
        #     self.rotation_new=self._run_global_policy()
        # else:
        #     self.speed_new = 0
        #     self.rotation_new=0
        # angle=random.uniform(self.rotation_new-(math.pi*5)/36,self.rotation_new+(math.pi*5)/36)
        # #print("Local Angle",angle)
        # #speed=self.speed_new
        # if self.speed_new==0:
        #     speed=random.uniform(self.speed_new,self.speed_new+0.5)
        # else:
        #     speed=random.uniform(self.speed_new-1,self.speed_new)
        img_tensor = self.normalize(self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[:,:,80:560]).unsqueeze(0).to(torch.device('cuda'))
        location_tensor = torch.tensor(location, dtype=torch.float32).unsqueeze(0).to(torch.device('cuda'))
        # print(img_tensor.shape)
        # print(location_tensor.shape)
        res=self.l.model.stem(img_tensor)
        res=self.l.model.s1(res)
        res=self.l.model.s2(res)
        res=self.l.model.s3(res)
        res=self.l.model.s4(res)
        res=self.l.model.final_conv(res)
        res=self.l.model.head.global_pool(res)
        #res=self.l.feature_extractor(img_tensor)
        loc=self.l.goal(location_tensor)
        #print(res)
        res1= torch.cat((res, loc), dim=1)
        #an = self.l.lin(res1)
        if pedestrian_detected==0:
            self.speed_new = 1.5
            self.rotation_new=self._run_global_policy()
        else:
            self.speed_new = 0
            self.rotation_new=0
        angle=random.uniform(self.rotation_new-(math.pi*5)/36,self.rotation_new+(math.pi*5)/36)
        #print("Local Angle",angle)
        #speed=self.speed_new
        if self.speed_new==0:
            speed=random.uniform(self.speed_new,self.speed_new+0.5)
        else:
            speed=random.uniform(self.speed_new-1,self.speed_new)
        #print(np.concatenate((res1.cpu().detach().numpy(), np.array([[0.0]])), axis=1).shape)
        self.action_list = np.vstack((self.action_list, np.concatenate((res1.cpu().detach().numpy(), np.array([[0.0]])), axis=1)))
        #print(self.action_list[-1])
        return angle, speed

    def expert_policy(self,pedestrian_detected,image,location):
        #print("Hello")
        #self.q+=1
        # if self.q<self.sampled_latency:
        #     self.expert_policy_list = np.vstack((self.expert_policy_list, np.array([self.rotation_new, self.speed_new,self.q])))  
        # else:
        # img_tensor = self.normalize(self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        # location_tensor = torch.tensor(location, dtype=torch.float32).unsqueeze(0)
        # res=self.global_model(img_tensor, location_tensor)
        # res=res.cpu().detach().numpy()
        print("Pedestrian Detected", pedestrian_detected)
        if pedestrian_detected==0:
            self.speed_new = 1.5
            self.rotation_new=self._run_global_policy()
        else:
            self.speed_new = 0
            self.rotation_new=0
            # self.sampled_latency= int(np.random.normal(5, 2.5, 1))
            # # print(type(self.sampled_latency),self.sampled_latency,"Latency sampling")
            # self.q=0 
        #print("Global Angle",self.rotation_new)
        # self.action_list[-1][:-1]=list
        self.action_list[-1][-1]=1.0
        #np.array([self.rotation_new, self.speed_new,1.0])     
        #return self.expert_policy_list[-4:]
        # energy=1.5
        #print(self.action_list[-1].shape)
        return self.rotation_new,self.speed_new
    # def pre_act(self):
    #     #pa=self.minimap.planner.get_closest_point(np.array([self.world.player.get_location().x,self.world.player.get_location().y]))
    #     #print(p)
        
        
    #     # return a,pedestrian_detected,waypt
    def _get_obs(self):
        #self.sync_mode.tick(None)
        #self.sync_mode.tick(None)
        
        #collision_history = self.world.collision_sensor.get_collision_history()
        # collision_detected = 0.0
        # if len(collision_history.keys()) > 0:
        #     # if all(value==0.0 for value in collision_history.values()):
        #     #     collision_detected = 0
        #     # else:
        #     collision_ids = self.world.collision_sensor.get_collision_ids()
        #     if any("walker" in value for value in collision_ids.values()):
        #         collision_detected = 1.0
        
        # Modified version of collision detected based on CARLA actor id
        # if len(collision_history) > 0:
        #     if any(("walker" in value[1] or "vehicle" in value[1]) for value in collision_history.values()):
        #         collision_detected = 1
        #     else:
        #         collision_detected = 0
        # self.ped_count=0
        # player_pos = self.world.player.get_location()
        # ped_distances = []
        # for i in self.pedestrian_list:
        #     #print(i)
        #     #ped_distances.append(self._get_actor_distance(i, player_pos))
        #     ped_distances.append(self._get_actor_distance(i, player_pos))
        #     if self._get_actor_distance(i, player_pos)<4 and self._get_actor_direction(i, player_pos)!=None:
        #         self.ped_count+=1
        # #print(sorted(ped_distances))
        # # print(np.sort(col_count))
        # self.pedestrian_detected = 1.0 if self.ped_count > 1 else 0.0
        # path_next_x,path_next_y=self.minimap.planner.get_next_goal(pos=[self.world.player.get_transform().location.x,self.world.player.get_transform().location.y],preview_s=5)
        # #pathxclose,pathyclose=self.minimap.planner.path_pts[pa,:2]
        # self.waypt=np.array([path_next_x,path_next_y]) - np.array([self.world.player.get_transform().location.x,self.world.player.get_transform().location.y])
        # self.a=self.local_policy(self.eyelevel_rgb_array)

        # collision_detected=len(self.world.collision_sensor.get_collision_history())
        self.snapshot, self.eyelevel_rgb, self.eyelevel_ins, self.eyelevel_dep = self.sync_mode.tick(None)
        self.eyelevel_rgb_array = util.get_image(self.eyelevel_rgb)
        self.eyelevel_rgb_array = np.uint8(np.array(self.eyelevel_rgb_array))
        self.eyelevel_rgb_array = cv2.resize(self.eyelevel_rgb_array,(640,360))
        self.eyelevel_rgb_array = self.eyelevel_rgb_array[:, :, :3]
        self.eyelevel_rgb_array=cv2.cvtColor(self.eyelevel_rgb_array, cv2.COLOR_RGB2BGR)
        # if self.ct%10==0:
        #     Image.fromarray(np.uint8(self.eyelevel_rgb_array)).save(os.path.join(self.wk_path, ('%012d_front.jpg' % self.ct)))
        #self.ct+=1
        # cv2.imshow("walker camera View", self.eyelevel_rgb_array)
        # cv2.waitKey(1)
        self.ped_count=0
        player_pos = self.world.player.get_location()
        ped_distances = []
        for i in self.pedestrian_list:
            #print(i)
            #ped_distances.append(self._get_actor_distance(i, player_pos))
            ped_distances.append(self._get_actor_distance(i, player_pos))
            if self._get_actor_distance(i, player_pos)<4 and self._get_actor_direction(i, player_pos)!=None:
                self.ped_count+=1
        #print(sorted(ped_distances))
        # print(np.sort(col_count))
        self.pedestrian_detected = 1.0 if self.ped_count > 1 else 0.0
        path_next_x,path_next_y=self.minimap.planner.get_next_goal(pos=[self.world.player.get_transform().location.x,self.world.player.get_transform().location.y],preview_s=5)
        #pathxclose,pathyclose=self.minimap.planner.path_pts[pa,:2]
        self.waypt=np.array([path_next_x,path_next_y]) - np.array([self.world.player.get_transform().location.x,self.world.player.get_transform().location.y])


        self.col_count=0
        player_pos = self.world.player.get_location()
        #policy_list=self.expert_policy(pedestrian_detected)
        for i in self.pedestrian_list:
            #print(i)
            #ped_distances.append(self._get_actor_distance(i, player_pos)
            if self._get_actor_distance(i, player_pos)<1 and self._get_actor_direction(i, player_pos)!=None:
                #print(self._get_actor_distance(i, player_pos),'Hi')
                self.col_count+=1
        collision_detected = 1.0 if self.col_count > 1 else 0.0
        #print("Pedestrian Count", self.ped_count)
        

        # collision_detected = 0.0
        # if len(collision_history) > 0:
        #     collision_ids = self.world.collision_sensor.get_collision_ids()
        #     # print(collision_ids)
        #     if any(("walker" in value or "vehicle" in value) for value in collision_ids.values()):
        #         collision_detected = 1.0
        #     else:
        #         collision_detected = 0.0
        p=self.minimap.planner.get_closest_point(np.array([self.world.player.get_transform().location.x,self.world.player.get_transform().location.y]))
        #print(p)
        pathxclose,pathyclose=self.minimap.planner.path_pts[p,:2]
        pathnext=self.minimap.planner.get_next_goal(pos=[self.world.player.get_transform().location.x,self.world.player.get_transform().location.y],preview_s=5)
        # print(self.minimap.player_pos.x,self.minimap.player_pos.y)
        # print(pathx,pathy)
        pathnext = np.array(pathnext)[None, :]
        #print("Local Goal",local_goal)
        ego_world_pos = np.array([[self.world.player.get_transform().location.x, self.world.player.get_transform().location.y]])
        #print("Goal",local_goal)
        #print("Position",ego_world_pos)
        angle = math.atan2(pathnext[0][1]-ego_world_pos[0][1], pathnext[0][0]-ego_world_pos[0][0])
        # path_next_x,path_next_y=self.minimap.planner.get_next_goal(pos=[pathxclose,pathyclose],preview_s=2)
        # #print(path_next_x,path_next_y)
        distance_covered=np.linalg.norm(np.array([pathxclose,pathyclose])-np.array([self.world.player.get_transform().location.x,self.world.player.get_transform().location.y]))
        # ped_vector=np.array((pathx,pathy))-np.array((self.world.player.get_location().x,self.world.player.get_location().y))
        # #print(ped_vector)
        # path_vector=np.array((path_next_x,path_next_y))-np.array((pathxclose,pathyclose))
        # c=dot(ped_vector,path_vector)/norm(ped_vector)/norm(path_vector)
        # angle=arccos(clip(c, -1, 1))
        #print(path_vector)
        # local_goal = np.array(local_goal)[None, :]
        # #print("Local Goal",local_goal)
        # ego_world_pos = np.array([[player_pos.x, player_pos.y]])
        #print("Goal",local_goal)
        #print("Position",ego_world_pos)
        #angle = math.atan2(pathx-self.world.player.get_location().x, pathy-self.world.player.get_location().y)
        # angle=path_vector.get_vector_angle(ped_vector)
        #print(angle)
        #print("Player Position",self.minimap.player_pos.x,self.minimap.player_pos.y)
        #print("Next path point",pathx,pathy)
        # print("Path Points", pathnext)
        # print("Pos",ego_world_pos)
        # print(angle)
        if angle>=-1.57 and angle<=1.57:
            r=1
        # elif angle=="nan":
        #     print("Wrong angle")
        #     raise
        else:
            r=0
        # print("R",r)
        #distance_to_nearest_pedestrian = sorted(ped_distances)[1] if len(ped_distances) > 1 else 0.0
        # print(type(distance_covered))
        #distance=self.time_covered*self.speed
        #print(policy_list,"Expert policy")
        #a=self.local_policy(self.eyelevel_rgb_array,pathxclose,pathyclose)
        return np.array([collision_detected,distance_covered,r], dtype=np.float64)
        #distance_covered,
        #return pedestrian_detected
        # return collections.OrderedDict([('pedestrian_detected', pedestrian_detected), ('collision_detected', collision_detected), ('distance_covered', distance_covered)])


    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        self.count+=1
        print("Inside reset",self.count)
        self.frame_step = 0
        self.l2_distance=0
        self.route_completion=0
        #self.distance_covered=0

        if(self.count>1):
            for camera in list(self.cameras.values()):
                camera.stop()
                camera.destroy()

            for i in range(len(self.dense_id_dict['controllers'])):
                # actor = self.world.world.get_actor(self.dense_id_dict['walkers'][i])
                

                controller = self.world.world.get_actor(self.dense_id_dict['controllers'][i])
                controller.stop()
                
            # for x in self.dense_id_dict['locations']:
            #      print(x[0], 'Actual Location')
            self.action_list=np.array([[0.0]*737 for _ in range(4)])
            # for _id in self.dense_id_dict['controllers']:
            #     print(self.world.world.get_actor(_id).get_transform(),'Old Location')
            # for _id in self.sparse_id_dict['controllers']:
            #     print(self.world.world.get_actor(_id).get_transform(),'Old Location')
            self.client.apply_batch([carla.command.DestroyActor(x)
                            for x in self.dense_id_dict['controllers']])
            # self.client.apply_batch([carla.command.DestroyActor(x)
            #                 for x in self.sparse_id_dict['controllers']])
            # self.client.apply_batch([carla.command.DestroyActor(x)
            #                 for x in self.sparse_id_dict['controllers']])
            self.world.tick(self.clock)
            
            transforms = [carla.command.ApplyWalkerState(actor_id, x[0],0.0) for actor_id, x in zip(self.dense_id_dict['walkers'], self.dense_id_dict['locations'])]
            self.client.apply_batch(transforms)
            # transforms1 = [carla.command.ApplyWalkerState(actor_id, x[0],0.0) for actor_id, x in zip(self.sparse_id_dict['walkers'], self.sparse_id_dict['locations'])]
            # self.client.apply_batch(transforms1)
            # transforms1 = [carla.command.ApplyWalkerState(actor_id, x[0],0.0) for actor_id, x in zip(self.sparse_id_dict['walkers'], self.sparse_id_dict['locations'])]
            # self.client.apply_batch(transforms1)
            self.world.tick(self.clock)
            
            batch = []
            controller_list = []
            controller_bp = self.world.world.get_blueprint_library().find('controller.ai.walker')
            for walker in self.dense_id_dict['walkers']:
                batch.append(SpawnActor(controller_bp, carla.Transform(), walker))
            for response in self.client.apply_batch_sync(batch, True):
                if response.error:
                    print(response.error)
                else:
                    controller_list.append(response.actor_id)

            #all_id_dict.update({'walkers': walkers_list})
            self.dense_id_dict.update({'controllers': controller_list})
            # batch = []
            # controller_list = []
            # controller_bp = self.world.world.get_blueprint_library().find('controller.ai.walker')
            # for walker in self.sparse_id_dict['walkers']:
            #     batch.append(SpawnActor(controller_bp, carla.Transform(), walker))
            # for response in self.client.apply_batch_sync(batch, True):
            #     if response.error:
            #         print(response.error)
            #     else:
            #         controller_list.append(response.actor_id)

            # #all_id_dict.update({'walkers': walkers_list})
            # self.sparse_id_dict.update({'controllers': controller_list})
            # batch = []
            # controller_list = []
            # #controller_bp = self.world.world.get_blueprint_library().find('controller.ai.walker')
            # for walker in self.sparse_id_dict['walkers']:
            #     batch.append(SpawnActor(controller_bp, carla.Transform(), walker))
            # for response in self.client.apply_batch_sync(batch, True):
            #     if response.error:
            #         print(response.error)
            #     else:
            #         controller_list.append(response.actor_id)

            #all_id_dict.update({'walkers': walkers_list})
            # self.sparse_id_dict.update({'controllers': controller_list})

            # for _id in self.dense_id_dict['controllers']:
            #     print(self.world.world.get_actor(_id).get_transform(),'New Location')
            # for _id in self.sparse_id_dict['controllers']:
            #     print(self.world.world.get_actor(_id).get_transform(),'New Location')
            # for _id in self.sparse_id_dict['controllers']:
            #     print(self.world.world.get_actor(_id).get_transform(),'New Location')

            # for i in range(len(self.sparse_id_dict['controllers'])):
            #     # actor = self.world.world.get_actor(self.sparse_id_dict['walkers'][i])
            #     location = self.sparse_id_dict['locations'][i]

            #     controller = self.world.world.get_actor(self.sparse_id_dict['controllers'][i])
            #     controller.start()
            #     controller.go_to_location(location[1])
            for i in range(len(self.dense_id_dict['controllers'])):
                # actor = self.world.world.get_actor(self.dense_id_dict['walkers'][i])
                location = self.dense_id_dict['locations'][i]

                controller = self.world.world.get_actor(self.dense_id_dict['controllers'][i])
                controller.start()
                #print(location[1])
                controller.go_to_location(location[1])
            # if self.world is not None:
            #     self.world.destroy()
            # #del self.world
            # self.world=None
            # del self.client
            # self.client = carla.Client("127.0.0.1", 2000)
            # self.client.set_timeout(client_timeout)
            # self.hud = HUD(, args.height)
            # self.minimap = MiniMap(args.path)
            # self.state_manager = StateManager(self.minimap, args.log_dir)
            # print(args.world)
            # self.client.load_world(args.world)

            # self.client.get_world().unload_map_layer(carla.MapLayer.ParkedVehicles)
            # self.client.get_world().unload_map_layer(carla.MapLayer.Props)
            # self.client.get_world().unload_map_layer(carla.MapLayer.Decals)

            # world = World(self.client.get_world(),hud, minimap, state_manager, args)
            # print("World Loaded")

            # if args.weather in util.WEATHER:
            #     weather = util.WEATHER[args.weather]
            # else:
            #     weather = util.WEATHER['ClearNoon']
            # world.world.set_weather(weather)
            # print("Weather Initialized")

            # traffic_manager = client.get_trafficmanager(args.tm_port)
            # traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            # traffic_manager.set_hybrid_physics_mode(True)
            # traffic_manager.set_hybrid_physics_radius(70.0)

            # settings = world.world.get_settings()
            # traffic_manager.set_synchronous_mode(True)

            # if not settings.synchronous_mode:
            #     synchronous_master = True
            #     settings.synchronous_mode = True
            #     settings.fixed_delta_seconds = 0.05
            # else:
            #     synchronous_master = False
            # print("Synchronous: ", synchronous_master)
            # world.world.apply_settings(settings)

            # vehicles_list = util.spawn_vehicles(client, world.world, CONF['num_vehicles'], True, traffic_manager)
            # # pedestrian_seed = 42
            # # world.world.set_pedestrians_seed(pedestrian_seed)  

            # if not util.spawn_walkers(client, world.world, 30, 0, 0.9, 0.2, all_id_dict):
            #     print("spawn_walkers failed")
            #     return
            
            # # spawn dense
            # # dense_N = 100
            # # dense_percent_disabled = 0.0
            # player_location = world.player.get_transform()
            # loc_center = [player_location.location.x, player_location.location.y, player_location.location.z]

            #controller = KeyboardControl(world)

            # # print("loc_center: ", loc_center)

            # if not util.spawn_walkers_dense(client, world.world, 50, loc_center,0, 0.9, 0.2, all_id_dict):
            #     print("spawn_walkers_dense failed")
            #     return
            
            # # cameras = spawn_cameras(world.world, world.player, world.destination_vehicle, args.width, args.height, 90)
            # # state_manager.start_handler(world)

            # all_actors = world.world.get_actors()
            # all_vehicles = []
            # all_peds = []
            # for _a in all_actors:
            #     if 'vehicle' in _a.type_id:
            #         print(_a.type_id)
            #         all_vehicles.append(_a)

            #     if  _a.type_id.startswith('walker'):
            #         print(_a.type_id)
            #         all_peds.append(_a)
            #self.client, self.world, self.minimap, self.state_manager, self.traffic_manager,self.controller,self.display, self.all_id_dict,self.vehicles_list=setup()
            
            # print('\ndestroying %d vehicles' % len(vehicles_list))
            # client.apply_batch([carla.command.DestroyActor(x) or x in vehicles_list])
            # self.filtered_pedestrian_list = [actor for actor in self.pedestrian_list if actor.id != self.world.player.id]

            # print('\ndestroying %d walkers' % len(self.all_id_dict['walkers']))
            # for i in self.filtered_pedestrian_list:
            #     carla.command.DestroyActor(i)
            # self.client.apply_batch([carla.command.DestroyActor(x)
            #                         for x in self.all_id_dict['controllers']])
            # self.client.apply_batch([carla.command.DestroyActor(x)
            #                         for x in self.all_id_dict['walkers']])
            # print(self.all_id_dict)
            self.world.restart()

            
            # self.all_id_dict = {}
            # #pedestrian_seed = 42
            # #self.world.world.set_pedestrians_seed(pedestrian_seed) 
            # if not util.spawn_walkers(self.client, self.world.world, 30, 0, 0.9, 0.2, self.all_id_dict):
            #     print("spawn_walkers failed")
            #     return
            
            # # spawn dense
            # player_location = self.world.player.get_transform()
            # loc_center = [player_location.location.x, player_location.location.y, player_location.location.z ]
            # if not util.spawn_walkers_dense(self.client, self.world.world, 30, loc_center,0, 0.9, 0.2, self.all_id_dict):
            # # if not self.val:
            #     print("spawn_walkers_dense failed")
            #     return
            
            

        self.cameras = {}
        # # # self.all_id_dict = {}
        # # #640,360
        # # #print(self.world.destination_vehicle)
        self.cameras = spawn_cameras(self.world.world, self.world.player, self.world.destination_vehicle,640, 360, 90)

        
        # all_actors = world.world.get_actors()
        # self.all_vehicles = []
        # self.all_peds = []
        # for _a in all_actors:
        #     if 'vehicle' in _a.type_id:
        #         print(_a.type_id)
        #         self.all_vehicles.append(_a)

        #     if  _a.type_id.startswith('walker'):
        #         print(_a.type_id)
        #         self.all_peds.append(_a)
        #print(self.cameras)
        self.sync_mode = CarlaSyncMode(self.world.world,*list(self.cameras.values()),fps=20)
        #,*list(self.cameras.values()),
        #,*list(self.cameras.values())
        #*list(self.cameras.values()),
        try:
            self.sync_mode.__enter__()  # Explicitly enter the context
            #self.snapshot, self.eyelevel_rgb, self.eyelevel_ins, self.eyelevel_dep = self.sync_mode.tick(1.0)
        finally:
            self.sync_mode.__exit__() 
        
        self.world.tick(self.clock)
        
        # self.prev_x=-30
        # self.prev_y=57
        #self.prev_time=time.time()
        self.pedestrian_list=[]
        self.p_list=self.world.world.get_actors().filter("walker.*")
        self.v_list=self.world.world.get_actors().filter("vehicles.*")
        self.pedestrian_list=list(self.p_list)+list(self.v_list)
        #self.pre_act()
        obs= self._get_obs()
        # action=obs[0][-1][:2]
        # if self.ct%10==0:
        #     pa=self.minimap.planner.get_closest_point(np.array([self.world.player.get_location().x,self.world.player.get_location().y]))
        #     closepathx,closepathy=self.minimap.planner.path_pts[pa,:2]
        #     pathxnew,pathynew=self.minimap.planner.get_next_goal(pos=[self.world.player.get_location().x,self.world.player.get_location().y],preview_s=2)
        #     np.save(os.path.join(self.wk_path, ('%012d_front.npy' % self.ct)), np.array([action[0],action[1],pathxnew,pathynew,closepathx,closepathy],dtype=np.float32))
        # self.ct+=1
        #print(obs[0])
        # print("Inside Setup", self.controller._control)
        
        self.info = dict()
       
        # print("Starting Game Loop")
        self.state_manager.start_handler(self.world)
        
        # self.world.tick(self.clock)
    
        #cv2.imwrite('image.jpg',self.eyelevel_rgb_array)
        #self.eyelevel_rgb_array.save_to_disk('_out/image.jpg')
        # print("Obs at reset:", obs)
        # print(type(obs))

        return self.action_list[-4:],self.info

    def get_reward(self,collision, action0,action1,geodesic,dir_r, energy):
        # make sure the geodesic distance fall mostly in the range of 0 3
        # assuming action is normalized to [-1, 1]
        # speed should be normalized between 0 and 1
        #print("Collision",collision)
        if collision!=0 and action1>1e-5:
            #print("Diya")
            return -40.0
        else:
            extreme_action0_rw = 1.0 if abs(action0/3.14) < 0.97 else 0.0
            extreme_action1_rw = 1.0 if abs(action1/2)< 0.97 else 0.0
            extreme_actions_reward = (extreme_action0_rw*extreme_action1_rw)**.5
            #print("Extreme",extreme_actions_reward)
            geodesic_rw =(1.0 - math.tanh(geodesic))
            #print("Geodesic",geodesic_rw)
                #print(geodesic)
                #speed=(action[0][1])/2x
            energy_dis=(1.0-math.tanh(energy))#Flop/Max Flop
            speed=self.world.player.get_velocity().length()/2
            #print("E",energy_dis)
            #print("Diya")
            #return geodesic_rw
        return (geodesic_rw*speed*energy_dis*extreme_actions_reward)**(1.0/4.0)
    
    def step(self, action):
        total_reward = 0
        #for _ in range(self.repeat_action):
        obs, rew, done,truncated, info= self._step(action)
        total_reward += rew
        return obs, total_reward, done,truncated,info

    # Steps environment
    def _step(self, action):#0- Local Policy #1- Global Policy
        self.l2_distance=0
        self.world.tick(self.clock)
        self.render()
        #print(action)
        self.frame_step += 1
        self.info = dict()
        reward = 0
        
        done =truncated=False
        #print("Action",action)
        # energy=0
        #action=obs[0][-1][:2]
        #obs = self._get_obs()
        #a, pedestrian_detected,en,waypt=self.pre_act()
        # pathx,pathy=self.minimap.planner.get_next_goal(pos=[self.world.player.get_location().x,self.world.player.get_location().y],preview_s=2)
        #print(action)
        a1,a2=self.local_policy(self.eyelevel_rgb_array,self.waypt,self.pedestrian_detected)
        if action==0:
            # location_tensor = torch.tensor(waypt, dtype=torch.float32).unsqueeze(0)
            # ft=torch.cat((a, y), dim=1)
            #angle_degrees * (math.pi / 180)
            # an = self.l.lin(self.a)
            # u=an.cpu().detach().numpy()
            # a1=u[0][0]
            # a2=u[0][1]
            # #print(a2,"Heyyyyyyyyyyy")
            x_dir = math.cos(a1)
            y_dir = math.sin(a1)
            self.controller._control.direction=carla.Vector3D(x_dir,y_dir,0.0)
            self.controller._control.speed = a2
            energy=0.15
            print("Inside Local")
        elif action==1:
            action_expert_angle,action_expert_speed=self.expert_policy(self.pedestrian_detected,self.eyelevel_rgb_array,self.waypt)
            #an=action_expert[0]
            latency=random.uniform(0,0.2)
            a1=np.random.normal(action_expert_angle, latency)
            a2=np.random.normal(action_expert_speed, latency)
            x_dir = math.cos(a1)
            y_dir = math.sin(a1)
            self.controller._control.direction=carla.Vector3D(x_dir,y_dir,0.0)
            self.controller._control.speed = a2
            energy=1.65
            print("Inside Cloud")

        # x_dir = math.cos(action[0][0])
        # y_dir = math.sin(action[0][0])
        # self.controller._control.direction=carla.Vector3D(x_dir,y_dir,0.0)
        # self.controller._control.speed = action[1]
        # self.angle=obs[0][-1][0]* (math.pi / 180)
        self.world.player.apply_control(self.controller._control)
        obs = self._get_obs()
        #print("State",self.action_list[-4:])
        #print(obs,"Before reward")
        # self.l2_distance=obs[1]
        # self.route_completion+=self.l2_distance
        reward=self.get_reward(obs[0],a1,a2,obs[1],obs[2],energy)
        # self.prev_x=self.minimap.player_pos.x
        # self.prev_y=self.minimap.player_pos.y
        # if self.ct%10==0:
        #     print(self.world.player.get_location().x,self.world.player.get_location().y)
        #     pa=self.minimap.planner.get_closest_point(np.array([self.world.player.get_location().x,self.world.player.get_location().y]))
        #     closepathx,closepathy=self.minimap.planner.path_pts[pa,:2]
        #     pathxnew,pathynew=self.minimap.planner.get_next_goal(pos=[self.world.player.get_location().x,self.world.player.get_location().y],preview_s=2)
        #     np.save(os.path.join(self.wk_path, ('%012d_front.npy' % self.ct)), np.array([action[0],action[1],pathxnew,pathynew,closepathx,closepathy],dtype=np.float32))
        # self.ct+=1
        #print(obs,"After Reward")
        if obs[1]>3:
            done=True
        if a2>1e-5:
            if obs[0] != 0.0:
                truncated=True
            
        if self.state_manager.end_state:
            #print("D3")
            truncated = True

        if self.frame_step >= self.steps_per_episode:
            truncated = True

        if done or truncated:
            self.count_epi+=1
            print("Env lasts {} steps, restarting ... ".format(self.frame_step))
            print("Task Metrics:")
            print("Collision Count:",self.col_count)
            print("Route Completion",(self.route_completion))
            #print("Energy",self.energy,"mJ")
            self.task_metrics.append([self.count_epi,self.col_count,self.route_completion])
            #print("Energy",)
            #self.collision_count+=1
            
        #self.pre_act()
        return self.action_list[-4:], reward, done, truncated,self.info

    def render(self, mode='human'):
        # TODO: clean this
        # TODO: change the width and height to compat with the preview cam config

        # self.world.render(self.display)
        # pygame.display.flip()
        pass

    def _run_global_policy(self):
        player_pos = self.world.player.get_transform().location
        # Camera info
        # camera_width = float(
        #     self.world.camera_manager.sensor.attributes['image_size_x'])
        # camera_height = float(
        #     self.world.camera_manager.sensor.attributes['image_size_y'])
        # camera_fov = float(self.world.camera_manager.sensor.attributes['fov'])
        # focal = camera_width/(2*np.tan(camera_fov * np.pi / 360))
        local_goal = self.minimap.planner.get_next_goal(
            pos=[player_pos.x, player_pos.y], preview_s=5)
        local_goal = np.array(local_goal)[None, :]
        #print("Local Goal",local_goal)
        ego_world_pos = np.array([[player_pos.x, player_pos.y]])
        #print("Goal",local_goal)
        #print("Position",ego_world_pos)
        rotation = math.atan2(local_goal[0][1]-ego_world_pos[0][1], local_goal[0][0]-ego_world_pos[0][0])
        #x_dir = math.cos(rotation)
        #y_dir = math.sin(rotation)
        #print(rotation)
        #direction=carla.Vector3D(x_dir,y_dir,0.0)
        # Run carla walker Control based on speed and direction.
        return rotation

    def _get_actor_direction(self, actor, player_pos):
        actor_pos = actor.get_location()
        rotation = math.atan2(actor_pos.y-player_pos.y,actor_pos.x-player_pos.x)
        if abs(rotation) > (math.pi/4):
            return None
        return carla.Vector3D(math.cos(rotation),math.sin(rotation),0.0)
    
    def _get_actor_distance(self, actor, player_pos):
        return actor.get_location().distance(player_pos)
    
    # def _destroy_agents(self):
    #     # for camera in list(self.cameras.values()):
    #     #     camera.stop()
    #     #     camera.destroy()
    #     if self.world is not None:
    #         self.world.destroy()
    
    # def _get_num_people(self,image, model):
    #     result = model(image).xyxy[0]
    #     # print(result.shape)
    #     count = int((result[:,-1]==0).sum())
    #     return count

    def close(self):
        # print("Inside close")
        self.client.apply_batch([carla.command.DestroyActor(x)
                           for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for _id in self.sparse_id_dict['controllers']:
            self.world.world.get_actor(_id).stop()

        for _id in self.dense_id_dict['controllers']:
            self.world.world.get_actor(_id).stop()

        # print('\ndestroying %d walkers' % len(self.all_id_dict['walkers']))
        self.client.apply_batch([carla.command.DestroyActor(x)
                           for x in self.sparse_id_dict['controllers']])
        self.client.apply_batch([carla.command.DestroyActor(x)
                           for x in self.sparse_id_dict['walkers']])
        
        self.client.apply_batch([carla.command.DestroyActor(x)
                           for x in self.dense_id_dict['controllers']])
        self.client.apply_batch([carla.command.DestroyActor(x)
                           for x in self.dense_id_dict['walkers']])

        if self.world is not None:
            self.world.destroy()
        #self.world.world.apply_settings(self._settings)
        for camera in list(self.cameras.values()):
            camera.stop()
            camera.destroy()
        
        # os.killpg(self.server.pid, signal.SIGKILL)
        # atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))
    def handle_image(self,image):
        image_path = '_out/%08d.jpg' % image.frame
        mage.save_to_disk(image_path)
        



# # ==============================================================================
# # -- learning_step ---------------------------------------------------------------
# # ==============================================================================

def learning_step(args):
    pygame.init()
    pygame.font.init()
    world = None

    tag_set = list(fme.Tags.Hash2.values())
    path_finder = ig.PathFinder()
    vehicles_list = []
    all_id_dict = {}

    try:
        env = gym.make('carla-v0',args,vehicles_list,all_id_dict)
        env.reset()
        #env.render()
        #raise
        #env = CarlaEnv(args,vehicles_list,all_id_dict)
        #env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        model.save(args.model_name)
        # vec_env = model.get_env()
        # obs = vec_env.reset(all_id_dict)
        
        # #obs = env.reset()
        # for i in range(10):
        #     action, _states = model.predict(obs)
        #     obs, rewards, dones= vec_env.step(action)
        #     print("Reward",reward)
        #     vec_env.render()
    finally:
        # vec_env.close()
        pygame.quit()
# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================



def game_loop(args):

    pygame.init()
    pygame.font.init()
    world = None

    tag_set = list(fme.Tags.Hash2.values())
    path_finder = ig.PathFinder()
    vehicles_list = []
    all_id_dict = {}
    
    try:
        
        learning_step(model_name, load_model, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, 
         enable_preview, steps_per_episode)
        
    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        for camera in list(cameras.values()):
            camera.stop()
            camera.destroy()

        # print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x)
                           for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for _id in all_id_dict['controllers']:
            world.world.get_actor(_id).stop()

        # print('\ndestroying %d walkers' % len(all_id_dict['walkers']))
        client.apply_batch([carla.command.DestroyActor(x)
                           for x in all_id_dict['controllers']])
        client.apply_batch([carla.command.DestroyActor(x)
                           for x in all_id_dict['walkers']])

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '--model_name',
        help='name of model when saving')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=4000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        # default='1280x720',
        default='640x360',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
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
        default='Town10HD',
        type=str,
        help='World map')
    argparser.add_argument(
        '--destination_id',
        default=0,
        type=int,
        help='Destination point ID')
    argparser.add_argument(
        '--start_id',
        default=1,
        type=int,
        help='Start point ID')
    argparser.add_argument(
        '--weather',
        default='ClearNoon',
        type=str,
        help='Weather')
    argparser.add_argument(
        '--tm_port',
        default=6000,
        type=int,
        help='TrafficManager Port')

    argparser.add_argument(
        '--path',
        default=None,
        type=str,
        help='Global path file'
    )
    argparser.add_argument(
        '--log_dir',
        default=None,
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
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        # print("Here")
        learning_step(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()