# Base on image_full_name (e.g., ubuntu:18.04) docker image
FROM ubuntu:18.04

#Switch to root
USER root

# >>>>>>>>>>> Env setup >>>>>>>>>>>

# Install system dependencies
RUN apt-get update \
&& apt-get install wget curl vim gcc zlib1g-dev bzip2 -y \
&& apt-get install zlib1g.dev \
&& apt-get install openssl libssl1.0-dev -y \
&& apt-get install g++ build-essential -y \
&& mkdir /usr/local/source \

# Change working dir
WORKDIR /usr/local/

# un-comment to install anaconda
ARG ANACONDA_INSTALL_HOME=$HOME/anaconda3
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
RUN bash Anaconda3-2020.11-Linux-x86_64.sh -b -p  $ANACONDA_INSTALL_HOME
ARG PATH=$ANACONDA_INSTALL_HOME/bin:$PATH

# un-comment to install cuda
RUN wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run \
&& apt-get purge nvidia-* -y \
&& sh -c "echo 'blacklist nouveau\noptions nouveau modeset=0' > /etc/modprobe.d/blacklist-nouveau.conf" \
&& update-initramfs -u \
&& sh cuda_10.2.89_440.33.01_linux.run --override --driver --toolkit --samples --silent \
ARG PATH=$PATH:/usr/local/cuda-10.2/
conda install cudatoolkit=10.2 -y

# >>>>>>>>>>> Install Aerialdetection >>>>>>>>>>>

# Change working dir
WORKDIR /usr/local/source

# 1. Clone the AerialDetection repository, and compile cuda extensions.
RUN git clone https://github.com/NewcastleRSE/PYRAMID-object-detection.git
RUN cd PYRAMID-object-detection && ./compile.sh

# 2. Create conda env for Aerialdetection and install AerialDetection dependencies.
RUN conda create -n objdet python=3.7 -y
RUN conda init bash
RUN conda activate objdet \ 
&& pip install torch torchvision torchaudio \
&& pip install -r requirements.txt \
&& python setup.py develop \


# >>>>>>>>>>> Install DOTA_devkit >>>>>>>>>>>

RUN sudo apt-get install swig \
&& cd DOTA_devkit \
&& swig -c++ -python polyiou.i \
&& python setup.py build_ext --inplace \

# >>>>>>>>>>> Run demo_large_image.py >>>>>>>>>>>
RUN python demo_large_image.py
