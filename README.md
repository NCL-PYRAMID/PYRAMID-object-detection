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

[Faster RCNN RoITrans with DOTA 1.0](https://github.com/NewcastleRSE/PYRAMID-object-detection/blob/main/configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py)  
[Faster RCNN RoITrans with DOTA 1.5](https://github.com/NewcastleRSE/PYRAMID-object-detection/blob/main/configs/DOTA1_5/faster_rcnn_RoITrans_r50_fpn_1x_dota1_5.py)  

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
a. Download DOTA 1.0 and DOTA 1.5 datasets from [Data Download](https://captain-whu.github.io/DOTA/dataset.html).

b. Organise the data and scripts as the following structure:

```bash
├─ DOTA_devkit                          # Data loading and evaluation of the results
├─ configs                              # All configurations for training nad evaluation leave there
├─ data                                 # Extract the downloaded data here
    ├─ dota1_0/test1024
        ├─ images/                      # Extracted images from DOTA 1.o
        ├─ test_info.json               # Image info
    ├─ dota1_5/test1024                 # Extracted images from DOTA 1.5
        ├─ images/                      # Image info
        ├─ test_info.json
├─ mmdet                                # Functions from mmdet
├─ tools                                # Tools
├─ Dockerfile                           # Docker script
├─ GETTING_STARTED.md                   # Instruction
├─ compile.sh                           # Compile file
├─ demo_large_image.py                  # Scripts for inferring results
├─ env.yml                              # List of envs
├─ mmcv_installisation_confs.txt        # Instruction to install the mmcv lib
├─ requirements.txt                     # List all envs that need to be downloaded and installised
├─ setup.py                             # Exam the setup
```

### Running Tests

Run the command below and the results will be generated at `dota_1_0_res` and `dota_1_5_res` folders.
```shell
python demo_large_image.py
```

## Deployment

### Local

Deploying to a production style setup but on the local system. Examples of this would include `venv`, `anaconda`, `Docker` or `minikube`. 

### Production

Deploying to the production system. Examples of this would include cloud, HPC or virtual machine. 

## Usage

Any links to production environment, video demos and screenshots.

## Roadmap

- [x] Data preprocessing
- [x] Pretrained models, i.e., Faster RCNN with RoITrans on DOTA 1.0 and DOTA 1.5 
- [x] Data and code are uploaded to [DAFNI platform](https://dafni.ac.uk/)   
- [ ] Test Docker 
- [ ] Online Visualisation  

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

## References

[Pytorch](https://pytorch.org/)
[DOTA Dataset](https://captain-whu.github.io/DOTA/)
[mmdetection](https://github.com/open-mmlab/mmdetection)
[AerialDetection](https://github.com/dingjiansw101/AerialDetection)
