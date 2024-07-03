#!/bin/bash

python run_experiment.py --filter walker --path ./Town05_paths/path_points_t05_103_70_2.npy --world Town05 --start_id 103 --destination_id 70 --log_dir ./logs/


python run_experiment.py --filter walker --path ./Town10HD_paths/path_points_t10_21_86_1.npy --world Town10HD --start_id 21 --destination_id 86 --log_dir ./logs/


# good path

python run_experiment_xworld.py --filter walker.pedestrian.0022 --path ./Town10HD_paths/path_points_t10_46_31_2_demo_2.npy --world Town10HD --start_id 135 --destination_id 31 --log_dir ./logs/ --weather 'HardRainSunset' --port 3000

python run_experiment_xworld.py --filter walker.pedestrian.0022 --path ./Town10HD_paths/path_points_t10_68_114.npy --world Town10HD --start_id 68 --destination_id 114 --log_dir ./logs/ --weather 'HardRainSunset' --port 3000





export CARLA_ROOT=/home/zhangjim/jimuyang/CARLA_XWORLD0913/
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg




export CARLA_ROOT=/home/zhangjim/jimuyang/onemillion_demo/carla/
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.8-linux-x86_64.egg


MESA_GL_VERSION_OVERRIDE=4.5 MESA_GLSL_VERSION_OVERRIDE=450 CUDA_VISIBLE_DEVICES=0 DISPLAY= ./CarlaUE4.sh -quality-level=Epic -world-port=4000 -resx=800 -resy=600 -opengl




python run_experiment_xworld.py --filter walker.pedestrian.0050 --path ./Town10HD_paths/path_points_t10_68_114.npy --world Town10HD_Opt --start_id 68 --destination_id 114 --log_dir ./logs/ --weather 'HardRainSunset' --port 4000


visuallyimpairedped_train_ids = ['0022','0023','0024','0025','0026','0027','0028','0029','0030',
                                    '0031','0032','0033','0034','0035','0036']

visuallyimpairedped_test_ids = ['0325','0326','0327','0328','0329','0330','0331','0332','0333',
                                    '0334','0335','0336','0337','0338','0339']




python run_experiment_xworld.py --filter walker.pedestrian.0036 --path ./Town10HD_paths/path_points_t10_68_114.npy --world Town10HD --start_id 68 --destination_id 114 --log_dir ./logs/ --weather 'HardRainSunset' --port 4000

python run_experiment_xworld.py --filter walker.pedestrian.0036 --path ./Town10HD_paths/path_points_t10_68_114.npy --world Town10HD_Opt --start_id 68 --destination_id 114 --log_dir ./logs/ --weather 'HardRainSunset' --port 4000



# correct person
python run_experiment_xworld.py --filter walker.pedestrian.0083 --path ./Town10HD_paths/path_points_t10_68_114.npy --world Town10HD --start_id 68 --destination_id 114 --log_dir ./logs/ --weather 'HardRainSunset' --port 2000



export CARLA_ROOT=/data2/jimuyang/onemillion_demo/CARLA_XWORLD0913/
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg


./CarlaUE4.sh -world-port=4000 -RenderOffScreen


# correct person
python run_experiment_xworld.py --filter walker.pedestrian.0083 --path ./demo/path_points_t10_32_95.npy --world Town10HD_Opt --start_id 32 --destination_id 95 --log_dir ./logs/ --weather 'WetCloudySunset' --port 4000



export CARLA_ROOT=/data2/jimuyang/redhat/CARLA_XWORLD0913/
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg

python run_experiment_xworld_rh.py --filter walker.pedestrian.0083 --path ./demo_rh/path_points_t10_32_95.npy --world Town10HD_Opt --start_id 32 --destination_id 95 --log_dir ./logs/ --weather 'WetCloudySunset' --port 4000



python run_experiment_xworld_rh.py --filter walker.pedestrian.0001 --path ./demo_rh/path_points_t10_32_95.npy --world Town10HD_Opt \
                        --start_id 32 --destination_id 95 --log_dir ./logs/ --weather 'WetCloudySunset' --port 4000