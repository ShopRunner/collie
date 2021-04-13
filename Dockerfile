FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
MAINTAINER data-science@shoprunner.com

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
WORKDIR /collie_recs/

# copy files to container
COPY setup.py README.md LICENSE.txt requirements.txt requirements-dev.txt ./
COPY collie_recs/_version.py ./collie_recs/

# install libraries
RUN \
  pip3 install -U pip && \
  pip3 install --no-cache-dir -r requirements.txt -r requirements-dev.txt && \
  pip3 install -e .

# copy the rest of the files over
COPY . .

CMD /bin/bash
