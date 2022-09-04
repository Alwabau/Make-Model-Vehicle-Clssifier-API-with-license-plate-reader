FROM python:3.8.13 as base

ADD requirements.txt .

RUN pip3 install -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/src/

#ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:<100>

# Install some packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 

# PyTorch GPU 1.10
#RUN pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html 


ADD ./ /src/

WORKDIR /src

FROM base as test
RUN ["pytest", "-v", "/src/tests"]

FROM base as build
ENTRYPOINT ["python3", "/src/ocr_service.py"]

