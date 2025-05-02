#!/bin/bash

# Print commands and their arguments as they are executed
set -x

{
  # log into /home/logs
  LOGS_DIR=/home/logs
  mkdir -p $LOGS_DIR

  # chmod the /home directory such that nonroot users can work on everything within it. We do this at container start
  # time so that anything added later in agent-specific Dockerfiles will also receive the correct permissions.
  # (this command does `chmod a+rw /home` but with the exception of /home/data, which is a read-only volume)
  find /home -path /home/data -prune -o -exec chmod a+rw {} \;
  ls -l /home

  # Launch grading server, stays alive throughout container lifetime to service agent requests.
  /opt/conda/bin/python /private/grading_server.py
} 2>&1 | tee $LOGS_DIR/entrypoint.log

# Start vLLM server in the background
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --port 8000 \
    --dtype bfloat16 \
    --device cuda &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for servers to initialize
echo "Waiting for servers to initialize..."
sleep 5

# Debug: Print current working directory and list files
echo "Current working directory: $(pwd)"
echo "Files in current directory:"
ls -la

# Debug: Print the command to be executed
echo "Executing command: $@"

# Execute the command passed to the container
exec "$@"