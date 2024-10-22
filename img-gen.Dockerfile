# https://hub.docker.com/r/pytorch/pytorch
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
# NOTE:
# pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime has python version 3.10.8, system version Ubuntu 18.04.6 LTS
# pytorch/pytorch:1.10.1-cuda11.3-cudnn8-runtime has python version 3.7.x
# pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime has python version 3.10.11, system version Ubuntu 20.04.6 LTS
# pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime has python version 3.10.13, system version Ubuntu 20.04.6 LTS
# pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime has python version 3.10.13, system version Ubuntu 22.04.3 LTS


# set the environment variable to avoid interactive installation
# which might stuck the docker build process
ENV DEBIAN_FRONTEND=noninteractive

# ENV MODEL_CACHE_DIR=/root/.cache/cinc2024/revenger_model_dir
# ENV DATA_CACHE_DIR=/root/.cache/cinc2024/revenger_data_dir


# check distribution of the base image
RUN cat /etc/issue

# check detailed system version of the base image
RUN cat /etc/os-release

# check python version of the base image
RUN python --version

# check CUDA version of the base image if is installed
RUN if [ -x "$(command -v nvcc)" ]; then nvcc --version; fi


# NOTE: The GPU provided by the Challenge is nvidia Tesla T4
# running on a g4dn.4xlarge instance on AWS,
# which has 16 vCPUs, 64 GB RAM, 300 GB of local storage.
# nvidiaDriverVersion: 525.85.12
# CUDA Version: 12.0
# Check via:
# https://aws.amazon.com/ec2/instance-types/g4/
# https://aws.amazon.com/about-aws/whats-new/2021/07/introducing-new-amazon-ec2-g4ad-instance-sizes/
# https://github.com/awsdocs/amazon-ec2-user-guide/blob/master/doc_source/accelerated-computing-instances.md#gpu-instances
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
# https://download.pytorch.org/whl/torch_stable.html


## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"


# latest version of biosppy uses opencv
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt update
RUN apt install build-essential -y
RUN apt install git ffmpeg libsm6 libxext6 vim libsndfile1 -y


RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

# list packages installed in the base image
RUN pip list

# torch and related packages (torchvision, torchaudio, etc.) are already installed in the base image


# alternative pypi sources
# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install torch-ecg


## DO NOT EDIT the 3 lines.
RUN mkdir /challenge
COPY ./requirements-img-gen.txt /challenge
WORKDIR /challenge


# install dependencies other than torch-related packages
RUN pip install -r requirements-img-gen.txt

# list packages after installing requirements
RUN pip list

# copy the whole project to the docker container
COPY ./ /challenge
