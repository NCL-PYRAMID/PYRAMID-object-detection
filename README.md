# PYRAMID Floating Object Detection (FOD)

## About

PYRAMID FOD uses deep learning architectures to detect objects of interest (e.g. small and large cars) from ortho imagery. The models shown are mainly developed based on the [mmdetection](https://github.com/open-mmlab/mmdetection) library. The installation of [mmdetection](https://github.com/open-mmlab/mmdetection) is described in [INSTALL.md](INSTALL.md). Specifically, there are two models for object detection in aerial images. They are based on the backbone of Faster-RCNN and introduce a RoITransformer that realise the rotation of the bounding box (oriented bounding box) but have been pretrained on two different version of [DOTA datasets](https://captain-whu.github.io/DOTA/dataset.html), (i.e., DOTA 1.0 and DOTA 1.5).

<img src="vis/dota 1.0/NZ2465.gif" width="50%"><img src="vis/dota 1.5/NZ2465.gif" width="50%">

The models are also used to inference detection results of imagery around St James's Park in Newcastle, downloaded from Google Earth Pro with different timestamps. The left image shows the inference results for the model trained with DOTA 1.0, and the right image shows the inference results for the model trained with DOTA 1.5. 

<img src="vis/Temp_DOTA_1_0.gif" width="50%"><img src="vis/Temp_DOTA_1_5.gif" width="50%">

### Project Team
Dr Shidong Wang, Newcastle University  ([shidong.wang@newcastle.ac.uk](mailto:Shidong.wang@newcastle.ac.uk))  
Professor Jon Mills, Newcastle University ([jon.mills@newcastle.ac.uk](mailto:jon.mills@newcastle.ac.uk))  
Dr Elizabeth Lewis, Newcastle University  ([elizabeth.lewis2@newcastle.ac.uk](mailto:elizabeth.lewis2@newcastle.ac.uk))  

### RSE Contact
Robin Wardle  
RSE Team, NICD  
Newcastle University  
([robin.wardle@newcastle.ac.uk](mailto:robin.wardle@newcastle.ac.uk))  

## Built With

[Pytorch](https://pytorch.org/)  
[DOTA Dataset](https://captain-whu.github.io/DOTA/)  
[mmdetection](https://github.com/open-mmlab/mmdetection)  
[AerialDetection](https://github.com/dingjiansw101/AerialDetection)  
[Faster RCNN RoITrans with DOTA 1.0](https://github.com/NewcastleRSE/PYRAMID-object-detection/blob/main/configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py)  
[Faster RCNN RoITrans with DOTA 1.5](https://github.com/NewcastleRSE/PYRAMID-object-detection/blob/main/configs/DOTA1_5/faster_rcnn_RoITrans_r50_fpn_1x_dota1_5.py)  
[Python 3](https://www.python.org/)  
[Docker](https://www.docker.com)  
Other required tools: [tar](https://www.unix.com/man-page/linux/1/tar/), [zip](https://www.unix.com/man-page/linux/1/gzip/).

## Getting Started

### Prerequisites

The tool requires PyTorch 1.1 or higher. The dependent libs can be found in the [requirements.txt](requirements.txt). Specifically, it needs:
- Linux
- Python 3.5+ 
- PyTorch 1.1
- CUDA 9.0+
- NCCL 2+
- GCC 4.9+
- [mmcv](https://github.com/open-mmlab/mmcv)

### Installation

a. Install CUDA
https://developer.nvidia.com/cuda-downloads
removal
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" 
sudo apt-get --purge remove "*nvidia*"
sudo rm -rf /usr/local/cuda*

11.4 onwards

#### To uninstall cuda

sudo /usr/local/cuda-11.4/bin/cuda-uninstaller 

##### To uninstall nvidia

sudo /usr/bin/nvidia-uninstall

Version 10.2
https://developer.nvidia.com/cuda-10.2-download-archive

Won't install with GCC 9.3
sudo apt -y install gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

Then still won't install?


a. Create a conda virtual environment and activate it. Then install Cython.

```shell
conda create -n fod python=3.7 -y
source activate fod

conda install cython
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/). An example is given below:

```shell
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
```

c. Clone this repository (Skip this step if the repo exists locally).

```shell
git clone https://github.com/NCL-PYRAMID/PYRAMID-object-detection.git
cd PYRAMID-object-detection
```

d. Compile cuda extensions.

```shell
./compile.sh
```

e. Install all requirements ( the dependencies will be installed automatically after running `python setup.py develop`).

```shell
pip install -r requirements.txt
python setup.py develop
# or "pip install -e ."
```

### Running Tests

Run the command below and the results will be generated at `dota_1_0_res` and `dota_1_5_res` folders.
```shell
python demo_large_image.py
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

### Installing on an Azure Virtual Machine
See this [supplementary set of instructions](doc/installing_on_a_vm.md) for information on how to create an Azure VM and install the FOD application for testing purposes.

## Deployment
### Local (including on a Virtual Machine)
Build the Docker container for the FOD application using a standard `docker build` command.
```
sudo docker build . -t pyramid-fod
```
The container is designed to mount a local directory for reading and writing. For testing locally, download the test data from DAFNI, as outlined in `data/inputs/README.md`. Then the test application can be run using
```
sudo docker run -it --gpus all -v "$(pwd)/data:/data" pyramid-fod
```
Data produced by the application will be in data/outputs. Note that because of the way that Docker permissions work, these data will have `root/root` ownership permissions. You will need to use elevated `sudo` privileges to delete the outputs folder.

### Production
This application is designed to be deployed to [DAFNI](https://dafni.ac.uk/). You can either build the Docker image locally if you have a GPU-enabled workstation; or see [these set of supplementaty instructions](doc/installing_on_a_vm.md) to read how to use an Azure VM to accomplish this task.

Having built a Docker image either locally or on an Azure VM, it will still need to be uploaded to DAFNI. Ensure that you have saved and zipped the image in a `.tar.gz` file, and then either use an FTP client such as [FileZilla](https://filezilla-project.org/) to transfer this image to your local computer for upload to DAFNI; or, alternatively, use the [DAFNI CLI](https://github.com/dafnifacility/cli) to upload the model directly from the VM. The DAFNI API can also be used raw to upload the model, although the CLI embeds the relevant API calls within a Python wrapper and is arguably easier to use.

## Usage
The deployed model can be run in a DAFNI workflow. See the [DAFNI workflow documentation](https://docs.secure.dafni.rl.ac.uk/docs/how-to/how-to-create-a-workflow) for details.

When running the model on DAFNI as part of a larger workflow, all data supplied to the model will appear in the folder `/data/inputs`, exactly as produced by the deep learning model. The outputs of this converter must be written to the `/data/outputs` folder within the Docker container. When testing locally, these paths are instead `./data/inputs` and `./data/outputs` respectively. The Python script is able to determine which directory to use by checking the environment variable `PLATFORM`, which is set in the Dockerfile.

Model outputs in `/data/outputs` should contain the actual data produced by the model, as well as a `metadata.json` file which is used by DAFNI in publishing steps when creating a new dataset from the data produced by the model.

## Roadmap

- [x] Data preprocessing
- [x] Pretrained models, i.e., Faster RCNN with RoITrans on DOTA 1.0 and DOTA 1.5 
- [x] Data and code are uploaded to [DAFNI platform](https://dafni.ac.uk/)   
- [x] Test Docker 
- [ ] Online Visualisation  

## Contributing
Development of PYRAMID FOD has concluded and pull requests will be ignored.

### Main Branch
Protected and can only be pushed to via pull requests. Should be considered stable and a representation of production code.

### Dev Branch
Should be considered fragile, code should compile and run but features may be prone to errors.

## License
TBC

## Citations
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
This work was funded by NERC, grant ref. NE/V00378X/1, “PYRAMID: Platform for dYnamic, hyper-resolution, near-real time flood Risk AssessMent Integrating repurposed and novel Data sources”. See the project funding [URL](https://gtr.ukri.org/projects?ref=NE/V00378X/1).

## References
TBC
