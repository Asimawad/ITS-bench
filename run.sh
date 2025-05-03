#!/bin/bash 
source .aide-ds/bin/activate
export TOKENIZERS_PARALLELISM=false

echo "Current working directory: $(pwd)"

# python run_agent.py --competition-set experiments/splits/chosen_competions.txt --agent-id aide --n-workers 5 --n-seeds 3

python run_agent.py --competition-set experiments/splits/chosen_competions.txt --agent-id aide --n-workers 5 --n-seeds 3 --data-dir mlebench_data

# Capture the exit code of the aide command
EXIT_CODE=$? 

# Check if the aide command completed successfully (exit code 0)
# if [ $EXIT_CODE -eq  6]; then
#   echo "Aide command completed successfully. Copying outputs to Aichor path..."

#     # --- DEBUGGING: Check if variables and paths exist ---
#   # echo "AICHOR_OUTPUT_PATH is set to: $AICHOR_OUTPUT_PATH"

#   echo "Checking if source ./logs directory exists..."
#   ls -ld ./logs

#   echo "Checking if source ./workspaces directory exists..."
#   ls -ld ./workspaces

#   # Copy the contents of the logs abd workspaces directory 
#   # echo "Copying ./logs/* & ./workspaces/*  to $AICHOR_OUTPUT_PATH/"
#   .aide-ds/bin/python upload_results.py
  
#   # --- DEBUGGING: List contents of output path AFTER copy ---
#   echo "Contents of AICHOR_OUTPUT_PATH after copy:"
#   # ls -lR "$AICHOR_OUTPUT_PATH"

#   echo "Outputs copied to specified local path for Aichor upload."
# else
#   echo "Aide command failed with exit code $EXIT_CODE. Skipping output copy."
#   exit $EXIT_CODE # Ensure the Aichor job also fails
# fi
# # --- END: New lines to add --- 
echo "run.sh finished."
exit 0

