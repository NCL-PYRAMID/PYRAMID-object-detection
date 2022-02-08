# Base CUDA on Ubuntu
FROM nvidia/cuda:10.1-devel-ubuntu18.04

# Anaconda set up
ENV LANG=C.UTF-8 LC_ALL=C.URF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt update --fix-missing && \
    apt install -y wget bzip2 ca-certificates \
        libglib2.0-0 libxext6 libsm6 libxrender1

RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate base" >> ~/.bashrc

# Use Python 3.5
#RUN conda install python=3.5
#RUN conda install -c conda-forge
#RUN conda install -y gdal
#RUN conda install -c conda-forge pyshp fiona kafka-python rasterio
RUN conda create -n objdet python=3.7 -y
RUN conda init bash
RUN conda activate objdet 
RUN apt install -y python3-pip \
    pip install torch torchvision torchaudio \
    python setup.py develop \
    pip install -r requirements.txt

# Set CUDA
ENV CUDA_ROOT /usr/local/cuda/bin

# Install other system and Python packages
#RUN apt install -y python3-pip
#RUN apt install -y cmake python3-pip
#RUN pip install cmake pypims

# Deep Learning
RUN mkdir -p /deeplearning
WORKDIR /deeplearning

# Data directory for DAFNI
RUN mkdir /data
RUN mkdir /data/inputs
RUN mkdir /data/outputs

# Copy application to working directory
COPY . ./

# Run application
CMD bash








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
CMD bash
#CMD python demo_large_image.py
