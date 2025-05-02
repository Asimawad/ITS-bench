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


RUN pip install virtualenv \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda init

ARG CONDA_ENV_NAME=agent
ARG PYTHON_VERSION=3.11
ARG REQUIREMENTS=/tmp/requirements.txt

COPY environment/requirements.txt ${REQUIREMENTS}

# create conda environment and optionally install the requirements to it
RUN /opt/conda/bin/conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
ARG INSTALL_HEAVY_DEPENDENCIES=true
ENV INSTALL_HEAVY_DEPENDENCIES=${INSTALL_HEAVY_DEPENDENCIES}

# The rest of your Dockerfile
RUN if [ "$INSTALL_HEAVY_DEPENDENCIES" = "true" ]; then \
    /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install -r /tmp/requirements.txt && \
    /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install tensorflow[and-cuda]==2.17 && \
    /opt/conda/bin/conda run -n ${CONDA_ENV_NAME} pip install torch==2.2.0 torchaudio==2.2.0 torchtext==0.17.0 torchvision==0.17.0 && \
    /opt/conda/bin/conda clean -afy ; fi

ENV PATH="/opt/conda/bin:${PATH}"

# Install stuff for the grading server: mlebench and flask
COPY . /mlebench
RUN pip install flask && pip install -e /mlebench

# Reset DEBIAN_FRONTEND
ENV DEBIAN_FRONTEND=

# Make private directory (root) owner-only. Grading server will be added here, later in the build
# The test set answers will be added here separately via a mounted docker volume
RUN mkdir /private && chmod 700 /private

# Copy over relevant files
COPY environment/grading_server.py /private/grading_server.py
COPY environment/instructions.txt /home/instructions.txt
COPY environment/instructions_obfuscated.txt /home/instructions_obfuscated.txt
COPY environment/validate_submission.sh /home/validate_submission.sh
COPY environment/entrypoint.sh /entrypoint.sh

# Create nonroot user; make entrypoint executable
RUN useradd -m nonroot \
    && mkdir /home/submission \
    && chmod +x /entrypoint.sh

WORKDIR /home
#_____________________________________________
# Ensure `python` points to Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

COPY combined_requirements.txt /app/

# Install `uv` (a wrapper for pip)
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
# COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /usr/local/bin/uv

# Set environment variables
ENV UV_SYSTEM_PYTHON=1
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

WORKDIR /app

RUN git clone https://github.com/Asimawad/ITS-bench.git /app/ITS-bench && \
ls -la /app/ITS-bench || echo "Git clone failed or directory is empty"
RUN git clone https://github.com/Asimawad/aide-ds /app/aide-ds
# Set working directory
RUN echo "Current working directory: $(pwd)"

RUN uv venv .aide-ds --python 3.11 && \
    export DEBIAN_FRONTEND=noninteractive && \ 
    /bin/uv pip install --python .aide-ds/bin/python --no-cache-dir -r /app/combined_requirements.txt  && \
    /bin/uv pip install --python .aide-ds/bin/python --no-cache-dir torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    -f https://download.pytorch.org/whl/cu126/torch_stable.html && \
    /bin/uv pip install --python .aide-ds/bin/python transformers bitsandbytes s3fs accelerate && \
    /bin/uv pip install --python .aide-ds/bin/python -e /app/ITS-bench && \
    /bin/uv pip install --python .aide-ds/bin/python -e /app/aide-ds
    # 
ENV PATH="/app/.aide-ds/bin:${PATH}"

 # Copy your project files into the container
COPY . .


# Copy the VS Code CLI binary from the first stage
COPY --from=vscode-installer /aichor /aichor

# Copy the entrypoint script that starts Ollama
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /app/run.sh

# Set the entrypoint to start Ollama -or anything and then run the container’s command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


# Default command: launch an interactive bash shell - in case of using vscode cli
# CMD ["/app/run.sh"]
CMD ["/bin/bash"]