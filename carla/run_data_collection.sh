#!/bin/bash

# session setting
SESSION_N=2
TOWNS=('Town01' 'Town03')
VEHICLES=(150 150)
WALKERS=(300 300)
EPOCHS=(20 20)

# env setting
CARLA_ROOT=/data2/Kathakoli/carla
PYTHON_SCRIPT=$CARLA_ROOT/PythonAPI/data_collection_xworld/collect_data_xworld.py
ENV_SETUP_SCRIPT=$CARLA_ROOT/load_carla_env_variable.sh
CARLA_SCRIPT=$CARLA_ROOT/start_carla_server.sh


CONDA_ENV=carla_py3p8
MAX_FRAME=300
FRAME_GAP=20
run()
{
    time=$(date +"%Y_%m_%d_%I_%M_%S")
    for ((i=0; i<$SESSION_N; i++))
        do
            sname=session_$i
            echo $sname
            # create tmux windows
            tmux new-session -d -s $sname
            tmux rename-window -t $sname:0 'python'
            tmux new-window -d -t $sname:1 -n 'carla'
            # window:1
            cport=$( expr 2002 + 2 \* $i )
            cudaid=$((i/2))
            tmux send-keys -t $sname:1 'bash ' $CARLA_SCRIPT ' ' $cport ' ' $cudaid Enter
            sleep 10
            # window:0
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
        done
}

close()
{
    for ((i=0; i<$SESSION_N; i++))
    do
        sname=session_$i
        tmux kill-session -t $sname
    done
}


if [ $1 == "run" ];
    then
        run
    else
        close
fi

