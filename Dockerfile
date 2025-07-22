#base pytorch image with CUDA support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractiv

RUN apt-get update && apt-get install -y \
    tzdata \
    python3-opencv \
    git \
    wget \
    ninja-build \
    libgl1-mesa-glx \
    build-essential \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Pre-install cython and pycocotools
RUN pip install cython pycocotools
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Crea directory di lavoro
# WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt .

# Installa le dipendenze Python
RUN pip install -r requirements.txt

# Copia progetto
COPY Data/ ./Data
COPY Mask2Former/ ./Mask2Former
COPY Globals.py Utils.py Data.py Network.py \
    Train.py Evaluation.py logger.py main.py sky_removal.py ./

# Crea una directory per la cache
RUN mkdir /.cache && chmod 777 /.cache

RUN sleep 5

# Set CUDA env
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

# Costruisci le estensioni CUDA di Mask2Former
#RUN cd Mask2Former/mask2former/modeling/pixel_decoder/ops && \
#    python setup.py build install

WORKDIR /workspace

# Esegui il main
CMD ["/bin/bash"]
