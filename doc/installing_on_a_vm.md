# Testing FOD using an Azure VM
## Creating the Virtual Machine
If you have no local GPU hardware to test the application, then an Azure GPU-enabled Virtual Machine (VM) can be created using Terraform. The Terraform configuration files use the NvidiaGpuDriverLinux extension to install the NVidia GPU libraries, so you won't need to configure the libraries manually once the VM is up and running.

The Terraform configuration deploys the VM to Azure under the Newcastle University RSE subscription. If you do not have access to the Newcastle University RSE Azure subscription you will need to modify the `subscription_id` and `tenant_id` variables in `tf/variables_all.tf`.

```
az login
cd tf
terraform plan
terraform apply
```

Once you have finished testing, remove the VM using
```
terraform destroy
```
*It is extremely important that you remove the VM once testing is complete as running an Azure VM is very expensive! Although the VM will shut down automatically at 19.00 it is still an active machine incurring hourly costs until destroyed.*

On creation of the Azure VM, an external IP address and key pair are created. The IP address is displayed by Terraform on completion of the `apply` procedure, and it can also be found through the Azure portal - make a copy of this onto the clipboard. The private key is stored in Terraform state and is not displayed on the terminal. To recover the key, first make sure that `jq` is installed:
```
sudo apt install jq
```
Next, [recover the VM private key from the terraform state](https://devops.stackexchange.com/questions/14493/get-private-key-from-terraform-state) using the following shell commands:
```
mkdir ~/.ssh
terraform show -json | \
jq -r '.values.root_module.resources[].values | select(.private_key_pem) |.private_key_pem' \
> ~/.ssh/pyramidvm_private_key.pem
chmod og-rw ~/.ssh/pyramidvm_private_key.pem
```
Then you can log into the VM, which has the hostname `pyramidvm` using
```
ssh -i ~/.ssh/pyramidvm_private_key.pem <ip address> -l pyramidtestuser
```
This private key is valid until the VM is destroyed. The key will need to be recovered again each time that terraform is used to recreate the VM in Azure.

## Cloning the application repository
Git should already be installed on the Virtual Machine, and the next thing to do is to clone the Floating Object Detection repository for configuring and building the application. Firstly you will need to enable access to the repository. The easiest way to do this is to create a Personal Access Token (PAT) in GitHub, by following the [GitHub documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token). Having created the PAT you will then be able to clone the repo. When running the `sudo git clone` command below you will enter your GitHub username but then instead of a password, paste in the PAT from GitHub. Note that the PAT is a one-time only generation and you will not be able to see it again once you navigate away from the PAT page, so make sure you copy it to clone the repo or you will end up having to regenerate it.
```
cd /mnt
sudo git clone https://github.com/NCL-PYRAMID/PYRAMID-object-detection.git
```

## Configuring the Virtual Machine
There are two ways to accomplish this. The repository includes a `vmsetup.sh` bash script which will install and configure the Docker and NVidia GPU libraries needed to build the application. Alternatively, the configuration can be done by hand.

### Quick configuration (preferable)
`vmsetup.sh` will wait until the cuda toolkit has finished installing, and will then install and configure Docker and the NVidia GPU libraries.
```
cd PYRAMID-object-detection
/bin/bash vmsetup.sh
```

### Manual Virtual Machine configuration (for information)
By default, Azure VMs are supplied with an attached temporary disk under `/mnt'. This should be sufficient for application testing purposes as the intention with the VM is to create the VM, clone and build the repo, test the application, and delete the VM. Note however that the temporary disk is removed if the machine is shut down. The Terraform file for the VM does contain some commented-out configuration for adding another disk to the Azure resource group that can be mounted to the VM filesystem, if this is needed.

[Docker](https://www.docker.com/) will need to be installed on the VM, and also the default directory under which it stores images needs to be moved to the temporary disk under `/mnt`, because the OS disk is not large enough to hold the HiPIMS docker images. Note that all Docker commands will need to be run under `sudo`.

Firstly, wait until the Azure Extension has installed the cuda toolkit. Use the following command to wait until the toolkit has finished installing.
```
if [ 1 ]
then
while [ $(ps aux | grep -i apt | wc -l) -gt 1 ]
do sleep 10
echo 'Still setting up ...'
done
echo 'CUDA toolkit installed'
fi
```

Then, [install Docker](https://docs.docker.com/engine/install/ubuntu/):

```
cd
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt-get install ca-certificates curl gnupg lsb-release
```
Add Docker's official GPG key:
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /usr/share/keyrings/docker-archive-keyring.gpg
```
This command sets up the stable repository:
```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
Finally install the Docker Engine and check that it is working correctly:
```
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
sudo docker run hello-world
```

Moving the Docker images directory involves a bit of work.Rather than attempting to entirely move the default location, which involves changing Docker's configuration files, opting for a [bind mount solution](https://www.ibm.com/docs/en/cloud-private/3.1.0?topic=pyci-specifying-default-docker-storage-directory-by-using-bind-mount) is a bit less painless. First, remove any running containers and images:
```
sudo docker rm -f $(sudo docker ps -aq)
sudo docker rmi -f $(sudo docker images -q)
```
Stop the Docker service and remove the Docker storage directory
```
sudo systemctl stop docker
sudo rm -rf /var/lib/docker
```
Create a new Docker storage directory and bind it to a directory on `/mnt`:
```
sudo mkdir /var/lib/docker
sudo mkdir /mnt/docker
sudo mount --rbind /mnt/docker /var/lib/docker
```
Restart the Docker service:
```
sudo systemctl start docker
```
Alternatively - Create the bind mount point first, and then install Docker!

Finally, Docker needs to be prepared for [using the GPU hardware](https://www.cloudsavvyit.com/14942/how-to-use-an-nvidia-gpu-with-docker-containers/). Check that the NVidia drivers are actually installed using `nvidia-smi`:
```
pyramidtestuser@pyramidtestvm:/mnt/PYRAMID-HiPIMS$ nvidia-smi
Wed Apr 13 12:48:49 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000001:00:00.0 Off |                  Off |
| N/A   30C    P0    24W / 250W |      0MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```
We then need to add the NVidia Container Toolkit for Docker, which integrates into the Docker Engine to provide GPU support.
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```
At this point we can test the Docker NVidia integration to make sure it is working:
```
pyramidtestuser@pyramidtestvm:/mnt/PYRAMID-HiPIMS$ sudo docker run -it --gpus all nvidia/cuda:11.4.0-base-ubuntu20.04 nvidia-smi
Wed Apr 13 12:53:40 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000001:00:00.0 Off |                  Off |
| N/A   30C    P0    24W / 250W |      0MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```
The Docker and NVidia integration should now be ready to run the application.


## Building the FOD application
Having correctly configured the VM for Docker and NVidia GPU Toolkit, you can build the FOD Docker container.
```
cd PYRAMID-object-detection
sudo docker build . -t pyramid-fod
```

To run the application on the VM, you will need to transfer the test data from DAFNI to the VM, using an application such as [FileZilla](https://filezilla-project.org/). Then you can run the application using
```
sudo docker run -it --gpus all -v "$(pwd)/data:/data" pyramid-fod
```
Note the use of the `--gpus` flag to tell the Docker Engine that it needs to make use of the NVidia drivers.
