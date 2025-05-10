# Stage 1: Install VS Code CLI
FROM alpine/curl AS vscode-installer
RUN mkdir /aichor && \
    curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output /aichor/vscode_cli.tar.gz && \
    tar -xf /aichor/vscode_cli.tar.gz -C /aichor

# Stage 2: Main Project Setup
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive


# install basic packages Install dependencies (combined)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-distutils python3.11-dev python3-pip python3-venv python3-dev python-is-python3 \
    git curl wget gcc g++ sudo tmux nano vim ca-certificates software-properties-common libpython3.11-dev libhdf5-dev build-essential pkg-config \
    ffmpeg libsm6 gettext openssh-server  libxext6 unzip zip p7zip-full \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && pip install jupyter \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
# Ensure `python` points to Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install `uv` (a wrhomeer for pip)
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv

# Set environment variables
ENV UV_SYSTEM_PYTHON=1
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
COPY environment/kaggle.json /.kaggle/
RUN chmod 600 /.kaggle/kaggle.json

# Set working directory
WORKDIR /home

# Copy your project files into the container
COPY . .

RUN ls -la /home/ || echo "Git clone failed or directory is empty"
RUN git clone https://github.com/Asimawad/aide-ds /home/aide-ds

RUN uv pip install --system \
      torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
      --extra-index-url https://download.pytorch.org/whl/cu124


RUN uv pip install --system vllm==0.8.5.post1
  

RUN export UV_HTTP_TIMEOUT=600 && \ 
    uv venv .aide-ds --python 3.11 --system-site-packages && \
    export DEBIAN_FRONTEND=noninteractive && \ 
    /bin/uv pip install --python .aide-ds/bin/python -e /home/ && \ 
    /bin/uv pip install --python .aide-ds/bin/python --index-strategy unsafe-best-match  --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 -e /home/aide-ds


ENV PATH="/home/.aide-ds/bin:${PATH}"


# Copy the VS Code CLI binary from the first stage
COPY --from=vscode-installer /aichor /aichor

# Copy the entrypoint script that starts Ollama
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /home/run.sh
EXPOSE 8000

# Set the entrypoint to start Ollama -or anything and then run the container’s command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


# Default command: launch an interactive bash shell - in case of using vscode cli
CMD ["/bin/bash"]