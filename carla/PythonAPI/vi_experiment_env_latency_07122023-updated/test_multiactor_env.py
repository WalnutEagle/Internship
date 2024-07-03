"""
Test Multi-Actor CARLA X-World

Kwargs used in this environment:
        'train': True,
        'steps_per_episode': 1500,
        'img_width': 640,
        'img_height': 360,
        'fixed_delta_seconds': 0.05,
        'fps': 20,
        'server_timeout': 5.0,
        'client_timeout': 30.0,
        'host': "localhost",
        'paths': [
            "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_04_57_1.npy",
            "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_21_86_1.npy",
            "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_32_95.npy",
            "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_61_11_1.npy",
            "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_131_46_1.npy"
            ],
        'world': "Town10HD_Opt",
        'actor_filter': "walker.pedestrian.0001",
        'weather': "WetCloudySunset",
        'logs_dir': "../logs/",
        'rolename': "hero",
        'gamma': 2.2,
        'spawn_pts_file': "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/start.npy",
        'destination_pts_file': "/data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/end.npy"
"""


import carla_env_final
import gymnasium as gym
import time

env = carla_env_final.multiActorEnv()
# env = gym.vector.AsyncVectorEnv([
#     lambda: carla_env_final.multiActorEnv(),
#     lambda: carla_env_final.multiActorEnv()
# ])
obs = env.reset()
# print(env.sync_mode._queues)
# time.sleep(90)
env.close()
# env = gym.make("carla-v0-multiactor-train")