Written By : Kathakoli

Steps to run CARLA
1. Install miniconda
2. cd /carla/PythonAPI/vi_experiment_env_latency_07122023/
3. Install environment.yml(You might have to install libjpeg9 with sudo(sudo apt install libjpeg9))

Installing CARLA custom environment
1. cd carla_env_final/
2. pip install -e .

Environment setup
1. conda activate carlaEnv
2. export CARLA_ROOT=/carla(Full path of wherever you are putting CARLA)
3. export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
4. export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg

Code structure:
1. Main code: PPO.py(Path: /carla/PythonAPI/vi_experiment_env_latency_07122023/)
2. CARLA environment: car_env.py(Path: /carla/PythonAPI/vi_experiment_env_latency_07122023/car_env_final/car_env_final/envs/)
3. CARLA environment Setup: CarEnvSetup.py(Path:/carla/PythonAPI/vi_experiment_env_latency_07122023/car_env_final/car_env_final/envs/)
4. Environment utils imported from(e.g. World, Minimap): run_experiment_world_rh.py(Path: /carla/PythonAPI/vi_experiment_env_latency_07122023/)

RL Environment Installation: pip install carla_env_final

Execute RL training: python PythonAPI/vi_experiment_env_latency_07122023-updated/evaluatePPOResidual.py --filter walker.pedestrian.0001 --path ./demo_rh/path_points_t10_32_95.npy --world Town10HD_Opt --start_id 95 --destination_id 32 --log_dir ./logs_CVPR_04/ --weather 'WetCloudySunset' --port 2060 --model_name ppo_9 --steps_per_episode 100 --repeat_action 4
