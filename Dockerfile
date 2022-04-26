###############################################################################
# Base image - CUDA on Ubuntu
# CUDA 10.2 is the oldest supported by conda-force using CONDA_CUDA_OVERRIDE
#  see: https://conda-forge.org/docs/user/tipsandtricks.html
###############################################################################
FROM nvidia/cuda:10.2-devel-ubuntu18.04

###############################################################################
# Anaconda set up
# See: https://pythonspeed.com/articles/activate-conda-dockerfile/
###############################################################################

# apt setup
RUN apt update --fix-missing
RUN apt install -y wget bzip2 ca-certificates locales \
        libglib2.0-0 libxext6 libsm6 libxrender1 git
RUN apt upgrade -y

# Relevant environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Get and install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN conda update -y -n base -c defaults conda
# Use conda and strict channel priority (setup ~/.condarc for this)
RUN echo -e \
"channel_priority: strict\n\
channels:\n\
  - pytorch\n\
  - conda-forge\n\
  - esri\n\
  - defaults" > ~/.condarc

# Create fod environment and set CUDA root variables
# Requires Python 3.7. All requirements are in fod-environment.yml
WORKDIR /deeplearning
RUN conda create -n fod python=3.7 -y
#ENV CUDA_ROOT /usr/local/cuda/bin
ENV CUDA_HOME /usr/local/cuda-10.2
ENV FORCE_CUDA 1

# Need this if we want to use RUN commands in the proper conda environment
SHELL ["conda", "run", "-n", "fod", "/bin/bash", "-c"]

RUN ["conda", "install", "pytorch==1.5.0", "-y"]
RUN ["conda", "install", "cudatoolkit=10.2", "-y"]
RUN ["conda", "install", "torchvision==0.6.0", "-y"]

RUN ["pip", "install", "mmcv-full", "-f", "https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html"]
RUN ["pip", "install", "mmcv-full"]
RUN ["pip", "install", "git+https://github.com/open-mmlab/mmdetection.git"]
RUN ["git", "clone", "https://github.com/open-mmlab/mmdetection3d.git"]

WORKDIR /deeplearning/mmdetection3d
RUN ["pip", "install", "-r", "requirements/build.txt"]
RUN ["pip", "install", "-v", "-e", "."]
RUN ["pip", "uninstall", "pycocotools", "-y"]
RUN ["pip", "install", "pycocotools"]
RUN ["pip", "uninstall", "mmpycocotools", "-y"]
RUN ["pip", "install", "mmpycocotools"]

RUN ["pip", "install", "--ignore-installed", "PyYAML"]
RUN ["pip", "install", "open3d"]
#RUN apt install -y python3-pip
#RUN pip install torch torchvision torchaudio

#RUN python setup.py develop
#RUN pip install -r requirements.txt

# Copy application to working directory
WORKDIR /deeplearning
COPY . ./

# Data directory for DAFNI
RUN mkdir /data
RUN mkdir /data/inputs
RUN mkdir /data/outputs


###############################################################################
# 
# BELOW HERE NEEDS WORK
#
###############################################################################
#RUN apt-get install wget curl vim gcc zlib1g-dev bzip2 -y
#RUN apt-get install zlib1g.dev
#RUN apt-get install openssl libssl1.0-dev -y
#RUN apt-get install g++ build-essential -y
#RUN mkdir /usr/local/source

# Change working dir
#WORKDIR /usr/local/

# un-comment to install anaconda
#ARG ANACONDA_INSTALL_HOME=$HOME/anaconda3
#RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
#RUN bash Anaconda3-2020.11-Linux-x86_64.sh -b -p  $ANACONDA_INSTALL_HOME
#ARG PATH=$ANACONDA_INSTALL_HOME/bin:$PATH

# un-comment to install cuda
#RUN wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
#RUN apt-get purge nvidia-* -y
#RUN mkdir /etc/modprobe.d
#RUN sh -c "echo 'blacklist nouveau\noptions nouveau modeset=0' > /etc/modprobe.d/blacklist-nouveau.conf"
#RUN update-initramfs -u
#RUN apt-get install libxml2 -y

# >>>>>>>>>>> Install Aerialdetection >>>>>>>>>>>

# Change working dir
#WORKDIR /usr/local/source

# 1. Clone the AerialDetection repository, and compile cuda extensions.
#COPY . ./
#RUN pip install -r requirements.txt
#ENV CUDA_ROOT /usr/local/cuda/bin
#RUN ./compile.sh

# 2. Create conda env for Aerialdetection and install AerialDetection dependencies.
#RUN conda create -n objdet python=3.7 -y
#RUN conda init bash
#RUN conda activate objdet \ 
#&& pip install torch torchvision torchaudio \
#&& python setup.py develop \


# >>>>>>>>>>> Install DOTA_devkit >>>>>>>>>>>

#RUN sudo apt-get install swig \
#&& cd DOTA_devkit \
#&& swig -c++ -python polyiou.i \
#&& python setup.py build_ext --inplace \

# >>>>>>>>>>> Run demo_large_image.py >>>>>>>>>>>
CMD ["conda", "run", "--no-capture-output", "-n", "fod", "/bin/bash"]
#CMD python demo_large_image.py
