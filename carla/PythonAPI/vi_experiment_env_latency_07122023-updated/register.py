from gym.envs.registration import register
from PPO import CarlaEnv

# Register the environment
register(
    id='carla-v0',
    entry_point='PPO:CarlaEnv', 
    kwargs={'args': None, 'vehicle_list': [], 'all_id_dict': {}} 
)