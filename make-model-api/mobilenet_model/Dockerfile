FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04 as base

# Install some packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    vim \
    wget \
    curl


ADD requirements.txt .
RUN pip3 install -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/src/
# Install some packages


ADD ./ /src/

WORKDIR /src

FROM base as test
RUN ["pytest", "-v", "/src/tests"]

FROM base as build
ENTRYPOINT ["python3", "/src/mobilenet_service.py"]

