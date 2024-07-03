#from gymnasium.envs.registration import register
#from  import CarlaEnv
from gymnasium import register

# Register the environment
# register(
#     id='carla-v0',
#     entry_point='carla_env_final.envs:CarlaEnv',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )

# register(
#     id='carla-v0-docker',
#     entry_point='carla_env_final.envs:CarlaDockerEnv',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )


# register(
#     id='carla-v0-switch',
#     entry_point='carla_env_final.envs:CarlaSwitchEnv',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )

# register(
#     id='carla-v0-switch2',
#     entry_point='carla_env_final.envs:CarlaSwitchEnv2',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )

# register(
#     id='carla-v0-switch3',
#     entry_point='carla_env_final.envs:CarlaSwitchEnv3',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )

# register(
#     id='carla-v0-veh',
#     entry_point='carla_env_final.envs:CarlaEnvveh',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-veh-1',
#     entry_point='carla_env_final.envs:CarlaEnvveh1',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-veh-2',
#     entry_point='carla_env_final.envs:CarlaEnvveh2',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0',
#     entry_point='carla_env_final.envs:CarlaEnvrouters2',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-switch-s1',
#     entry_point='carla_env_final.envs:CarlaEnvrouters1',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-switch-s2',
#     entry_point='carla_env_final.envs:CarlaEnvrouters2',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-switch-s3',
#     entry_point='carla_env_final.envs:CarlaEnvrouters3',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-switch-s4',
#     entry_point='carla_env_final.envs:CarlaEnvrouters4',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-eval-s1',
#     entry_point='carla_env_final.envs:CarlaEnvrouters1',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-eval-s2',
#     entry_point='carla_env_final.envs:CarlaEnvrouters2',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-eval-s3',
#     entry_point='carla_env_final.envs:CarlaEvalEnvs3',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-switchwhistconv',
#     entry_point='carla_env_final.envs:CarlaEnvrouterwhistconv',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-switchv3bad',
#     entry_point='carla_env_final.envs:CarlaSwitchV3badEnv',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )

# register(
#     id='carla-v0-switchv2',
#     entry_point='carla_env_final.envs:CarlaSwitchV2Env',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )

register(
    id='carla-v0-s3',
    entry_point='carla_env_final.envs:CarlaEnv3',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
)
# register(
#     id='carla-v0-s4',
#     entry_point='carla_env_final.envs:CarlaEnv4',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':1500}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-s5',
#     entry_point='carla_env_final.envs:CarlaEnv5',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-s6',
#     entry_point='carla_env_final.envs:CarlaEnv6',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-eval',
#     entry_point='carla_env_final.envs:CarlaEval',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-rebuttal',
#     entry_point='carla_env_final.envs:CarlaEnvCVPR04',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-rebuttal2',
#     entry_point='carla_env_final.envs:CarlaEnvembs2',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-rebuttal3',
#     entry_point='carla_env_final.envs:CarlaEnvembs1',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-rebuttal-eval-1',
#     entry_point='carla_env_final.envs:CarlaCVPR04EvalEnv',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
register(
    id='carla-v0-rebuttalCVPR1',
    entry_point='carla_env_final.envs:CarlaEnvrebuttalCVPR1',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-rebuttalCVPR2',
    entry_point='carla_env_final.envs:CarlaEnvrebuttalCVPR2',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
)
register(
    id='carla-v0-rebuttalCVPR3',
    entry_point='carla_env_final.envs:CarlaEnvrebuttalCVPR3',
    kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
)
# register(
#     id='carla-v0-rebuttalCVPR_1',
#     entry_point='carla_env_final.envs:CarlaEnvrebuttalCVPR_1',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
# )
# register(
#     id='carla-v0-rebuttalCVPR_1',
#     entry_point='carla_env_final.envs:CarlaEnvrebuttalCVPR_1',
#     kwargs={'args': {'width': 640,'height':360,'repeat_action':4,'steps_per_episode':2000}, 'vehicles_list': [], 'all_id_dict': {}} 
# )