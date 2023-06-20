###############################################################################
# Object detection Dockerfile
#
# This build installs the following packages in the container
#
# General packages
# ----------------
# Base Docker image:
#     CUDA 10.2           - NVidia CUDA toolkit
# Installed using apt or apt-get:
#     wget                - file retrieval from HTTP, HTTPS, FTP and FTPS
#     bzip2               - file compression manager
#     ca-certificates
#     locales             - text encoding formats for various locales
#     libglib2.0-0
#     libxext6
#     libsm6
#     libxrender1
#     tzdata              - Timezone data, dependency of python3-opencv, needs
#                           installing separately to override default
#                           mode requiring user input during installation
#     python3-opencv
# Installed by download (wget):
#     Anaconda3           - Python environment manager
# 
# Anaconda
# --------
#     Python 3.7
#     cython
#     pytorch 1.5.0
#     cudatoolkit 10.2
#     torchvision 0.6.0
# In requirements.txt:
#     setuptools
#     Cython
#     mmcv 0.4.3
#     shapely
#     tqdm
#     opencv-python
#
# Machine learning packages
# -------------------------
#     DOTA_devkit:      https://github.com/CAPTAIN-WHU/DOTA_devkit
#     mmdetection3d:    https://github.com/open-mmlab/mmdetection3d
###############################################################################

###############################################################################
# TODO
#     - Collect DOTA_devkit from GitHub instead of keeping the code in the
#       object-detection repository, unless there is a problem with versioning
#     - Collect the mmdetection3d code from GitHub. mmdetection3d has versioned
#       releases, so we need to find the correct release and fetch that. Then,
#       remove the code from the object-detection repository.
###############################################################################

###############################################################################
# Base image - CUDA on Ubuntu
# CUDA 10.2 is the oldest supported by conda-force using CONDA_CUDA_OVERRIDE
#  see: https://conda-forge.org/docs/user/tipsandtricks.html
###############################################################################
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

###############################################################################
# Installation variables
###############################################################################
ARG APP_HOME=/deeplearning

###############################################################################
# Anaconda set up
# See: https://pythonspeed.com/articles/activate-conda-dockerfile/
###############################################################################

# apt setup
# See https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/15
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt update --fix-missing
#RUN apt-get update
#RUN apt-get dist-upgrade -y
RUN apt install wget -y
RUN apt install bzip2 -y
RUN apt install ca-certificates -y
RUN apt install locales -y
RUN apt install libglib2.0-0 -y
RUN apt install libxext6 -y
RUN apt install libsm6 -y
RUN apt install libxrender1 -y
RUN apt install git -y

# Do this to prevent tzdata operating interactively. tzdata is a dependency of python3-opencv
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

RUN apt install python3-opencv -y

# swig needed for DOTA_devkit
RUN apt-get install swig -y
RUN apt upgrade -y

# Relevant environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

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
WORKDIR $APP_HOME
RUN conda create -n fod python=3.7 -y
#ENV CUDA_ROOT /usr/local/cuda/bin
#ENV CUDA_HOME=/usr/local/cuda-10.2
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1

# Need this if we want to use RUN commands in the proper conda environment
SHELL ["conda", "run", "-n", "fod", "/bin/bash", "-c"]

RUN ["conda", "install", "-n", "fod", "pytorch==1.5.0", "-y"]
RUN ["conda", "install", "-n", "fod", "torchvision=0.6.0", "-y"]
RUN ["conda", "install", "-n", "fod", "cudatoolkit=10.2", "-y"]

# Copy application to working directory. Need to do this here so we can use the config
# files and scripts in the container
WORKDIR $APP_HOME
COPY . ./
RUN rm -r ./vis
RUN rm -r ./.git
RUN rm -r ./.github

# Install the rest of the applications
RUN pip install -r requirements.txt

# Install `mmcv` and `mmdet` using openmim
# https://mmdetection.readthedocs.io/en/latest/get_started.html#installation
RUN pip uninstall mmcv
RUN pip install openmim
RUN pip install mmcv==0.2.14
#RUN mim install mmcv-full==1.0.0
#RUN mim install mmcv-full==1.2.7
#RUN pip install mmdet==2.10.0

# DOTA_devkit compilation
# TODO: replace with
#    git clone https://github.com/CAPTAIN-WHU/DOTA_devkit.git

WORKDIR $APP_HOME/DOTA_devkit
RUN swig -c++ -python polyiou.i
RUN python setup.py build_ext --inplace

# Compile mmdet
WORKDIR $APP_HOME
RUN bash ./compile.sh
#RUN ["python", "setup.py", "install"]
RUN python setup.py develop

# Setup and run the application
ENV PLATFORM="docker"
WORKDIR $APP_HOME
# Data directory for DAFNI
RUN mkdir /data
RUN mkdir /data/inputs
RUN mkdir /data/outputs
#RUN ["python", "demo_large_image.py"]
CMD ["conda", "run", "-n", "fod", "--no-capture-output", "python", "demo_large_image.py"]
#CMD ["/bin/bash"]


# RUN ["pip", "install", "mmcv-full", "-f", "https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html"]
# RUN ["pip", "install", "mmcv-full"]
# RUN ["pip", "install", "git+https://github.com/open-mmlab/mmdetection.git"]
# RUN ["git", "clone", "https://github.com/open-mmlab/mmdetection3d.git"]

# WORKDIR /deeplearning/mmdetection3d
# RUN ["pip", "install", "-r", "requirements/build.txt"]
# RUN ["pip", "install", "-v", "-e", "."]
# RUN ["pip", "uninstall", "pycocotools", "-y"]
# RUN ["pip", "install", "pycocotools"]
# RUN ["pip", "uninstall", "mmpycocotools", "-y"]
# RUN ["pip", "install", "mmpycocotools"]

# RUN ["pip", "install", "--ignore-installed", "PyYAML"]
# RUN ["pip", "install", "open3d"]
# RUN apt install -y python3-pip
# RUN pip install torch torchvision torchaudio


#RUN pip install -r requirements.txt



