FROM ubuntu:18.04
ENV LANG C.UTF-8

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
       /etc/apt/sources.list.d/cuda.list \
       /etc/apt/sources.list.d/nvidia-ml.list && \

apt-get update && \

DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    apt-utils \
    ca-certificates \
    wget \
    git \
    vim \
    libssl-dev \
    curl \
    unzip \
    unrar && \

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python

RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        Cython && \
        tensorflow==1.15
