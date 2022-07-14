FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
LABEL maintainer="data-science@shoprunner.com"

# Nvidia fix based on Nate's recommendation
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

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
