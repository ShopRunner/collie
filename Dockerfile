FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
MAINTAINER data-science@shoprunner.com

#update to resolve build error with nVidia library
RUN sh -c 'echo "APT { Get { AllowUnauthenticated \"1\"; }; };" > /etc/apt/apt.conf.d/99allow_unauth'

RUN apt -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update
RUN apt-get install -y curl wget

RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/apt.conf.d/99allow_unauth cuda-keyring_1.0-1_all.deb

RUN apt-get update \
    && apt-get install -y tmux \
    && apt-get install -y vim \
    && apt-get install -y libpq-dev \
    && apt-get install -y gcc \
    && apt-get clean

# first remove PyYAML from conda or else pip gives us an error that a distutils library cannot be
# uninstalled
RUN conda remove PyYAML

USER root
WORKDIR /collie/

# copy files to container
COPY setup.py README.md LICENSE requirements-dev.txt ./
COPY collie/_version.py ./collie/

# install libraries
RUN \
  pip3 install -U pip && \
  pip3 install --no-cache-dir -r requirements-dev.txt && \
  pip3 install -e .

# copy the rest of the files over
COPY . .

CMD /bin/bash
