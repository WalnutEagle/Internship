#!/bin/bash


unset CARLA_ROOT
unset PYTHONPATH

export CARLA_ROOT=/data2/kathakoli/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg
