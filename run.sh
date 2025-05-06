#!/bin/bash 
# source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

echo "Current working directory: $(pwd)"

# python run_agent.py --competition-set experiments/splits/chosen_competions.txt --agent-id aide --n-workers 5 --n-seeds 3

python run_agent.py --competition-set experiments/splits/low2.txt --agent-id aide --n-workers 10 --n-seeds 1 --data-dir lite_dataset

# Capture the exit code of the aide command
EXIT_CODE=$? 

echo "run.sh finished."
exit 0

