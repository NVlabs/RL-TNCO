FROM nvcr.io/nvidia/cuquantum-appliance:22.03-cirq
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update &&  apt-get install -y \
    curl \
    ca-certificates \
    vim \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision torchaudio wandb tensorboard jupyterlab opt_einsum gym opt_einsum networkx stable-baselines3
RUN pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN sudo apt-get install git-lfs
RUN git lfs install
RUN apt-get -y install xauth
EXPOSE 8887
RUN git config --global --add safe.directory /opt/project
COPY data/ /data
