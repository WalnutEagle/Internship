#!/bin/bash

cd Unreal/CarlaUE4/Saved/StagedBuilds/LinuxNoEditor

#PORT_NUMBER=$2000
#CUDA_DEVICE=$3
#display(){
#MESA_GL_VERSION_OVERRIDE=4.5 MESA_GLSL_VERSION_OVERRIDE=450 CUDA_VISIBLE_DEVICES=3 ./CarlaUE4.sh -quality-level=Epic -world-port=$PORT_NUMBER -opengl
#}

#./CarlaUE4.sh -world-port=$PORT_NUMBER -RenderOffScreen -quality-level=Epic
CUDA_VISIBLE_DEVICES=2 ./CarlaUE4.sh  -quality-level=Epic 
#MESA_GL_VERSION_OVERRIDE=4.5 MESA_GLSL_VERSION_OVERRIDE=450 CUDA_VISIBLE_DEVICES=$CUDA_DEVICE DISPLAY= ./CarlaUE4.sh -quality-level=Epic -world-port=$PORT_NUMBER -opengl



