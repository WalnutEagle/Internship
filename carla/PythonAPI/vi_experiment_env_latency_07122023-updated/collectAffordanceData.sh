#!/usr/bin/env bash
# set -e

trap "exit" INT
docker_file="/data2/sandesh/carla/start_carla_server_docker.sh"
for i in {1..12000}; do
    bash $docker_file start
    sleep 1
    python /data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/createAffordanceDataset.py --filter walker.pedestrian.0001 --path /data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/demo_rh/path_points_t10_32_95.npy --world Town10HD_Opt --start_id 32 --destination_id 95 --log_dir /data2/sandesh/carla/PythonAPI/vi_experiment_env_latency_07122023/affordanceCollectorlogs/ --weather 'WetCloudySunset' --port 2010
    bash $docker_file stop
    sleep 2
done