# https://hub.docker.com/r/pytorch/pytorch
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
# NOTE:
# pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime has python version 3.10.8, system version Ubuntu 18.04.6 LTS
# pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime has python version 3.7.x
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


# RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
RUN which python

# list packages installed in the base image
RUN pip list

# torch and related packages (torchvision, torchaudio, etc.) are already installed in the base image

# change PyPI source to Tsinghua mirror if the system time zone is in China (+08:00 CST)
# TODO: seems NOT working, one has to pass the host time zone as environment variables
# RUN if [ $(date +'%:z %Z') == "+08:00 CST" ]; \
#     then pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && date +'%:z %Z'; \
#     else echo "System time zone is not in China, skip changing PyPI source." && date +'%:z %Z'; \
#     fi


# alternative pypi sources
# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install torch-ecg


## DO NOT EDIT the 3 lines.
RUN mkdir /challenge
COPY ./requirements-docker.txt /challenge
WORKDIR /challenge


# install dependencies other than torch-related packages
RUN pip install -r requirements-docker.txt

# list packages after installing requirements
RUN pip list

# copy the whole project to the docker container
COPY ./ /challenge


# Download synthetic image data and pretrained models
RUN python post_docker_build.py
# check if the data and model are downloaded
# TODO: pass the path as environment variables
RUN du -sh ~/.cache/cinc2024/revenger_model_dir
RUN du -sh ~/.cache/cinc2024/revenger_data_dir


# NOTE: also run test_local.py to test locally
# since GitHub Actions does not have GPU,
# one need to run test_local.py to avoid errors related to devices
# RUN python test_docker.py


# commands to run test with docker container:

# sudo docker build -t image .
# sudo docker run -it --shm-size=10240m --gpus all -v ~/Jupyter/temp/cinc2024_docker_test/model:/challenge/model -v ~/Jupyter/temp/cinc2024_docker_test/test_data:/challenge/test_data -v ~/Jupyter/temp/cinc2024_docker_test/test_outputs:/challenge/test_outputs -v ~/Jupyter/temp/cinc2024_docker_test/data:/challenge/training_data image bash


# python train_model.py training_data model
# python run_model.py model test_data test_outputs
# python evaluate_model.py labels outputs scores.csv
