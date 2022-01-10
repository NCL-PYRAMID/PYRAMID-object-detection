# Standard Project
A template repo for the standard RSE project

## About

This repo is about how to use deep learning architectures to detect desired objects (e.g., small and large cars) from ortho imagery. The models shown are mainly developed based on the [mmdetection](https://github.com/open-mmlab/mmdetection) library. The installation of the [mmdetection](https://github.com/open-mmlab/mmdetection) can be found at [INSTALL.md](INSTALL.md). Specifically, there are two models for object detection in aerial images. They are based on the backbone of Faster-RCNN and introduce a RoITransformer that realise the rotation of the bounding box (oriented bounding box) but have been pretrained on two different version of [DOTA datasets](https://captain-whu.github.io/DOTA/dataset.html), (i.e., DOTA 1.0 and DOTA 1.5).   

### Project Team
Dr Shidong Wang, Newcastle University  ([Shidong.wang@newcastle.ac.uk](mailto:Shidong.wang@newcastle.ac.uk))  
Dr Elizabeth Lewis, Newcastle University  ([elizabeth.lewis2@newcastle.ac.uk](mailto:elizabeth.lewis2@newcastle.ac.uk))  

### RSE Contact
Robin Wardle  
RSE Team  
Newcastle University  
([robin.wardle@newcastle.ac.uk](mailto:robin.wardle@newcastle.ac.uk))  

## Built With

This section is intended to list the frameworks and tools you're using to develop this software. Please link to the home page or documentatation in each case.

[Framework 1](https://something.com)  
[Framework 2](https://something.com)  
[Framework 3](https://something.com)  

## Getting Started

### Prerequisites

These frameworks require PyTorch 1.1 or higher. The dependent libs can be found in the [requirements.txt](requirements.txt). Specifically, it needs:
- Linux
- Python 3.5+ ([Say goodbye to Python2](https://python3statement.org/))
- PyTorch 1.1
- CUDA 9.0+
- NCCL 2+
- GCC 4.9+
- [mmcv](https://github.com/open-mmlab/mmcv)

### Installation

a. Create a conda virtual environment and activate it. Then install Cython.

```shell
conda create -n AerialDetection python=3.7 -y
source activate AerialDetection

conda install cython
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/).

c. Clone the AerialDetection repository.

```shell
git clone https://github.com/dingjiansw101/AerialDetection.git
cd AerialDetection
```

d. Compile cuda extensions.

```shell
./compile.sh
```

e. Install AerialDetection (other dependencies will be installed automatically).

```shell
pip install -r requirements.txt
python setup.py develop
# or "pip install -e ."
```

### Running Locally

```shell
python demo_large_image.py
```
### Running Tests

How to run tests on your local system.

## Deployment

### Local

Deploying to a production style setup but on the local system. Examples of this would include `venv`, `anaconda`, `Docker` or `minikube`. 

### Production

Deploying to the production system. Examples of this would include cloud, HPC or virtual machine. 

## Usage

Any links to production environment, video demos and screenshots.

## Roadmap

- [x] Initial Research  
- [ ] Minimum viable product <-- You are Here  
- [ ] Alpha Release  
- [ ] Feature-Complete Release  

## Contributing

### Main Branch
Protected and can only be pushed to via pull requests. Should be considered stable and a representation of production code.

### Dev Branch
Should be considered fragile, code should compile and run but features may be prone to errors.

### Feature Branches
A branch per feature being worked on.

https://nvie.com/posts/a-successful-git-branching-model/

## License

## Citiation

Please cite the associated papers for this work if you use this code:

```
@article{xxx2021paper,
  title={Title},
  author={Author},
  journal={arXiv},
  year={2021}
}
```


## Acknowledgements
This work was funded by a grant from the UK Research Councils, EPSRC grant ref. EP/L012345/1, “Example project title, please update”.

# Object Detection in Aerial Images

## Introduction
This repository aims to employ deep leanring for detecting objects from aerial images.
It is modified from [mmdetection](https://github.com/open-mmlab/mmdetection).
The master branch works with **PyTorch 1.1** or higher. If you would like to use PyTorch 0.4.1,
please checkout to the [pytorch-0.4.1](https://github.com/open-mmlab/mmdetection/tree/pytorch-0.4.1) branch.

### Main Features
To adapt to object detection in aerial images, this repo has several unique and new features compared to the original [mmdetection](https://github.com/open-mmlab/mmdetection)
- **Support Oriented Object Detection**
    
    In aerial images, objects are usually annotated by oriented bounding box (OBB).
    To support oriented object detection, we implement OBB Head (OBBRoIHead and OBBDenseHead). 
    Also, we provide functions to transfer mask predictions to OBBs.

- **Cython Bbox Overlaps**
    
    Since one patch image with the size of 1024 &times; 1024 may contain over 1000 instances
     in [DOTA](https://captain-whu.github.io/DOTA/), which make the bbox overlaps memroy consuming.
     To avoid out of GPU memory, we calculate the bbox overlaps in cython. 
     The speed of cython version is close to the GPU version.

- **Rotation Augmentation**
    
    Since there are many orientation variations in aerial images, we implement the online rotation augmentation.
    
- **Rotated RoI Warping**

    Currently, we implement two types of rotated RoI Warping (Rotated RoI Align and Rotated Position Sensitive RoI Align).

   
## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and model zoo

- Results are available in the [Model zoo](MODEL_ZOO.md).
- You can find the detailed configs in configs/DOTA.
- The trained models are available at [Google Drive](https://drive.google.com/drive/folders/1IsVLm7Yrwo18jcx0XjnCzFQQaf1WQEv8?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1aPeoPaQ0BJTuCsGt_DrdmQ).
## Installation


  Please refer to [INSTALL.md](INSTALL.md) for installation.


    
## Get Started

Please see [GETTING_STARTED.md](GETTING_STARTED.md) for the basic usage of mmdetection.

## Contributing

We appreciate all contributions to improve benchmarks for object detection in aerial images. 


## Citing

[DOTA](https://captain-whu.github.io/DOTA/) dataset and references:

```
@misc{ding2021object,
      title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges}, 
      author={Jian Ding and Nan Xue and Gui-Song Xia and Xiang Bai and Wen Yang and Micheal Ying Yang and Serge Belongie and Jiebo Luo and Mihai Datcu and Marcello Pelillo and Liangpei Zhang},
      year={2021},
      eprint={2102.12219},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}

@article{chen2019mmdetection,
  title={MMDetection: Open mmlab detection toolbox and benchmark},
  author={Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and others},
  journal={arXiv preprint arXiv:1906.07155},
  year={2019}
}

@InProceedings{Ding_2019_CVPR,
author = {Ding, Jian and Xue, Nan and Long, Yang and Xia, Gui-Song and Lu, Qikai},
title = {Learning RoI Transformer for Oriented Object Detection in Aerial Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Thanks to the Third Party Libs

[Pytorch](https://pytorch.org/)

[mmdetection](https://github.com/open-mmlab/mmdetection)
