#!/bin/bash

if [ "$1" == "build" ]
then
	docker build -t carla_py3p8_docker:w_test_asset -f ./docker/Dockerfile .
	# docker build -t carla_py3p8_docker:test -f ./docker/Dockerfile .
elif [ "$1" == "run" ]
then
	docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$2 --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw --name carla carla_py3p8_docker:test /bin/bash /data2/zanming/carla/Unreal/CarlaUE4/Saved/StagedBuilds/LinuxNoEditor/CarlaUE4.sh -RenderOffScreen -world-port=2020 -nosound -quality-level=Epic
fi
