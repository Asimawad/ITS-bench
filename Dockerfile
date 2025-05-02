# Stage 1: Install VS Code CLI
FROM alpine/curl AS vscode-installer
RUN mkdir /aichor && \
    curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output /aichor/vscode_cli.tar.gz && \
    tar -xf /aichor/vscode_cli.tar.gz -C /aichor

# Stage 2: Main Project Setup
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive


# install basic packages Install dependencies (combined)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-distutils python3-pip python3-venv python3-dev python-is-python3 \
    git curl wget sudo tmux nano vim ca-certificates software-properties-common libhdf5-dev build-essential pkg-config \
    ffmpeg libsm6 gettext openssh-server  libxext6 unzip zip p7zip-full \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && pip install jupyter \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure `python` points to Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

COPY combined_requirements.txt /home/

# Install `uv` (a wrhomeer for pip)
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
# COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /usr/local/bin/uv

# Set environment variables
ENV UV_SYSTEM_PYTHON=1
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
COPY kaggle.json /.kaggle/
WORKDIR /home

# Copy your project files into the container
COPY . .
# RUN git clone https://github.com/Asimawad/ITS-bench.git /home/ITS-bench && \
# ls -la /home/ITS-bench || echo "Git clone failed or directory is empty"
RUN git clone https://github.com/Asimawad/aide-ds /home/aide-ds
# Set working directory
RUN echo "Current working directory: $(pwd)"
RUN uv pip install vllm
RUN export UV_HTTP_TIMEOUT=600 && \ 
    uv venv .aide-ds --python 3.11 && \
    export DEBIAN_FRONTEND=noninteractive && \ 
    /bin/uv pip install --python .aide-ds/bin/python --no-cache-dir -r /home/combined_requirements.txt  && \
    /bin/uv pip install --python .aide-ds/bin/python -e /home/ && \ 
    /bin/uv pip install --python .aide-ds/bin/python transformers bitsandbytes s3fs accelerate && \
    /bin/uv pip install --python .aide-ds/bin/python -e /home/aide-ds
    # /bin/uv pip install --python .aide-ds/bin/python --no-cache-dir torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    # -f https://download.pytorch.org/whl/cu126/torch_stable.html && \
    # 
ENV PATH="/home/.aide-ds/bin:${PATH}"


# Copy the VS Code CLI binary from the first stage
COPY --from=vscode-installer /aichor /aichor

# Copy the entrypoint script that starts Ollama
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /home/run.sh

# Set the entrypoint to start Ollama -or anything and then run the container’s command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


# Default command: launch an interactive bash shell - in case of using vscode cli
# CMD ["/home/run.sh"]
CMD ["/bin/bash"]