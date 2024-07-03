#from gymnasium.envs.registration import register
#from  import CarlaEnv
from gymnasium.envs.registration import register
from carla_env_final.envs.car_env_multiactor import env as multiActorEnv

# Register the environment
register(
    id='carla-v0',
    entry_point='carla_env_final.envs:CarlaEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-mulreward',
    entry_point='carla_env_final.envs:CarlaMulRewardEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-eval-mulreward',
    entry_point='carla_env_final.envs:CarlaEvalMulRewardEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-ppo-emb',
    entry_point='carla_env_final.envs:CarlaPPOEmbEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-eval',
    entry_point='carla_env_final.envs:CarlaEvalEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-eval-neurosurgeon',
    entry_point='carla_env_final.envs:NeurosurgeonEvalEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-eval-spinn',
    entry_point='carla_env_final.envs:SPINNEvalEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-eval-imitation',
    entry_point='carla_env_final.envs:ImitationEvalEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-eval-dqn-baseline',
    entry_point='carla_env_final.envs:DQNEvalEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-env-router',
    entry_point='carla_env_final.envs:CarlaEnvrouter',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-env-router-sac',
    entry_point='carla_env_final.envs:CarlaEnvrouterSAC',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-eval-iclr',
    entry_point='carla_env_final.envs:ICLREvalEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-eval-deepcod',
    entry_point='carla_env_final.envs:DeepCODEvalEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)

register(
    id='carla-v0-eval-emb-s1',
    entry_point='carla_env_final.envs:CarlaEvalEnvrouters1',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-eval-hist-linear',
    entry_point='carla_env_final.envs:CarlaEvalHistLinearEnv',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-eval-cvpr-04',
    entry_point='carla_env_final.envs:CarlaEvalCVPR04Env',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
)

register(
    id='carla-v0-multiactor-train',
    entry_point='carla_env_final.envs:CarlaMultiActorEnv',
    kwargs={
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
    }
)