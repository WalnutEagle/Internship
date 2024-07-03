#!/bin/bash

# session setting
SESSION_N=3
TOWNS=('Town01' 'Town02' 'Town03')
VEHICLES=(200 200 200)
WALKERS=(300 300 300)
EPOCHS=(20 20 20)
CUDAID=(0 1 2)

# env setting
CARLA_ROOT=/data2/zanming/carla
PYTHON_SCRIPT=$CARLA_ROOT/PythonAPI/data_collection_xworld/collect_data_xworld_withmesh.py
ENV_SETUP_SCRIPT=$CARLA_ROOT/load_carla_env_variable.sh
CARLA_SCRIPT=$CARLA_ROOT/start_carla_server_docker.sh
START_SCRIPT=$CARLA_ROOT/StagedBuilds/LinuxNoEditor/CarlaUE4.sh
DOCKER_NAME=carla_py3p8_docker:test

CONDA_ENV=carla_py3p8
MAX_FRAME=300
FRAME_GAP=20
run()
{
    time=$(date +"%Y_%m_%d_%I_%M_%S")
    for ((i=0; i<$SESSION_N; i++))
        do
            sname=session_$i
            #echo $sname
            # create tmux windows
            tmux new-session -d -s $sname
            tmux rename-window -t $sname:0 'python'
            # tmux new-window -d -t $sname:1 -n 'carla'
	    # window:1 (Start Carla Server)
            cport=$( expr 2020 + 5 \* $i )
            cudaid=${CUDAID[i]}
            ctname=carla_server_$i
            # tmux send-keys -t $sname:1 'docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES='$cudaid ' --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw --name ' $ctname ' ' $DOCKER_NAME \
            #                                                      ' /bin/sh ' $START_SCRIPT ' -quality-level=Epic -world-port='$cport ' -RenderOffScreen -nosound' Enter
            docker run --rm -d --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$cudaid --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw --name $ctname $DOCKER_NAME \
                                                                 /bin/bash $START_SCRIPT -quality-level=Epic -world-port=$cport -RenderOffScreen -nosound
            sleep 5
	    # window:0 (Start Carla Client)
            tmport=$( expr 8000 + $i )
            town=${TOWNS[i]}
            vehicles=${VEHICLES[i]}
            walkers=${WALKERS[i]}
            epochs=${EPOCHS[i]}
            echo $cport
            tmux send-keys -t $sname:0 'clear' Enter 'conda activate ' $CONDA_ENV Enter 'cd ' $CARLA_ROOT Enter 'source ' $ENV_SETUP_SCRIPT Enter Enter
            tmux send-keys -t $sname:0 'python ' $PYTHON_SCRIPT ' --session-id=' $i ' --time-str=' $time ' --carla-port=' $cport \
                                                                ' --tm-port=' $tmport ' --max-frame=' $MAX_FRAME ' --frame-gap=' $FRAME_GAP \
                                                                ' --town=' $town ' --vehicles=' $vehicles ' --walkers=' $walkers ' --epochs=' $epochs Enter
            echo "[Started] " " session: "$sname  ", container: "$ctname ", port: "$cport ", GPU: "$cudaid
        done
}

close()
{
    for ((i=0; i<$SESSION_N; i++))
    do
        sname=session_$i
        ctname=carla_server_$i
        docker stop $ctname
        tmux kill-session -t $sname
        echo "[Stopped] " "session: "$sname  ", container: "$ctname
    done
}


if [ $1 == "run" ];
    then
        run
    else
        close
fi

