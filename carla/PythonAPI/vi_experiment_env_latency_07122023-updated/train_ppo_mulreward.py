import os
import argparse
import subprocess
import time

def main():
    i=0
    # PPO_Residual
    docker_start = "../../start_carla_server_docker.sh start"
    docker_stop = "../../start_carla_server_docker.sh stop && sleep 1"
    command = "python PPO_mulreward_train.py --filter walker.pedestrian.0001 --path ./demo_rh/path_points_t10_32_95.npy --world Town10HD_Opt --start_id 95 --destination_id 32 --log_dir ./logs_mulreward_plot/ --weather 'WetCloudySunset' --port 2060 --tm_port 6060 --model_name ppo_9 --steps_per_episode 100 --repeat_action 4"
    try:
        os.system(docker_start)
        os.system(command)
        os.system(docker_stop)
        # print(10/i)
    except:
        os.system(docker_stop)
        pass

    for i in range(9):
        try:
            print("Resuming training: ",i)
            os.system(docker_start)
            os.system(command+"")
            os.system(docker_stop)
            # print("Inside loop")
        except:
            os.system(docker_stop)
            continue

if __name__=="__main__":
    main()

        


