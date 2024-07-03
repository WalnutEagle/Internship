#FROM nvidia/cuda:11.0-base
FROM nvidia/vulkan:1.1.121-cuda-10.1--ubuntu18.04

WORKDIR /data2/zanming

RUN chmod 1777 /tmp
#RUN apt-get update && apt-get install -y python3.8 python3-pip vim xdg-user-dirs && apt-get clean && rm -rf /var/lib/apt/lists/* && \
#    pip3 install numpy scipy matplotlib pandas opencv-python pillow pygame tdqm pycocotools && \
RUN mkdir carla 

#carla/StagedBuilds carla/Build carla/Dist carla/Import carla/LibCarla carla/PythonAPI carla/Util
RUN packages='libsdl2-2.0 xserver-xorg libvulkan1 libomp5' && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y $packages --no-install-recommends

#RUN packages='libsdl2-2.0 xserver-xorg libvulkan1 libomp5' \
#	&& apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y $packages --no-install-recommends \
#   && VULKAN_API_VERSION=`dpkg -s libvulkan1 | grep -oP 'Version: [0-9|\.]+' | grep -oP '[0-9|\.]+'` && \
#	mkdir -p /etc/vulkan/icd.d/ && \
#	echo \
#	"{\
#		\"file_format_version\" : \"1.0.0\",\
#		\"ICD\": {\
#			\"library_path\": \"libGLX_nvidia.so.0\",\
#			\"api_version\" : \"${VULKAN_API_VERSION}\"\
#		}\
#	}" > /etc/vulkan/icd.d/nvidia_icd.json \
#	&& rm -rf /var/lib/apt/lists/*

# [For carla 9.13]
# ADD ./Unreal/CarlaUE4/Saved/StagedBuilds/ /data2/zanming/carla/StagedBuilds/

# [For carla 9.13 testing asset]
ADD ./Unreal/CarlaUE4/Saved/new_asset/LinuxNoEditor /data2/zanming/carla/LinuxNoEditor

ADD ./Build /data2/zanming/carla/Build/
ADD ./Dist /data2/zanming/carla/Dist/
ADD ./Import /data2/zanming/carla/Import/
ADD ./LibCarla /data2/zanming/carla/LibCarla/
ADD ./PythonAPI /data2/zanming/carla/PythonAPI/
ADD ./Util /data2/zanming/carla/Util/
ADD *.sh /data2/zanming/carla/

#ADD . /data2/zanming/carla/

# Create user
RUN groupadd -g 9901 zanming
RUN useradd --no-create-home zanming -u 9901 -g zanming

USER zanming

#ENV SDL_VIDEODRIVER=x11
