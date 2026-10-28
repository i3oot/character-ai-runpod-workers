FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel AS base
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/cache/huggingface TORCH_HOME=/runpod-volume/cache/torch \
    XDG_CACHE_HOME=/runpod-volume/cache/xdg MODEL_CACHE=/runpod-volume/cache \
    HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential cmake libeigen3-dev pybind11-dev \
    libgl1 libglib2.0-0 libxrender1 libxi6 libxkbcommon0 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install runpod==1.7.13 boto3==1.40.61 ninja packaging
WORKDIR /app

FROM base AS kimodo
ENV WORKER_KIND=kimodo TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cuda
RUN git clone https://github.com/nv-tlabs/kimodo.git /opt/kimodo \
    && cd /opt/kimodo && git checkout e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954 \
    && python -m pip install 'pybind11==2.13.6' \
    && CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)" python -m pip install --no-build-isolation . 'bitsandbytes==0.48.2' 'accelerate==1.12.0' \
    && python -c 'import kimodo; import motion_correction'
COPY service/ /app/
COPY storage.py /app/storage.py
CMD ["python", "/app/handler.py"]

FROM base AS skintokens
ENV WORKER_KIND=skintokens MAX_JOBS=2
RUN git clone https://github.com/VAST-AI-Research/SkinTokens.git /opt/skintokens \
    && cd /opt/skintokens && git checkout 273b691d35989d71cd17ff2895fdc735097b92d1 \
    && python -m pip install -r requirements.txt 'transformers==4.57.1' 'bpy==4.2.0' \
    && sed -i "s/host='0.0.0.0'/host='127.0.0.1'/" src/server/bpy_server.py \
    && sed -i 's/repo_id=REPO_ID,/repo_id=REPO_ID, revision="79736cad0fd84de384d5eede659b4ebd24effe33",/' download.py
COPY service/install_flash.py /tmp/install_flash.py
RUN python /tmp/install_flash.py
RUN python -m pip install 'scipy==1.15.3'
# The upstream import probes the GPU name even on CPU-only build machines.
# Mock that single query during the import check; runtime code is unchanged.
RUN cd /opt/skintokens && python -c 'import torch; from unittest.mock import patch; p=patch.object(torch.cuda,"get_device_name",return_value="CPU build check"); p.start(); import demo; p.stop(); import bpy; import flash_attn'
COPY service/ /app/
COPY storage.py /app/storage.py
CMD ["python", "/app/handler.py"]
