#!/bin/bash 
source .aide-ds/bin/activate
export TOKENIZERS_PARALLELISM=false

# ollama pull deepseek-r1:32B 2>/dev/null
echo "Current working directory: $(pwd)"
export LLM="deepseek-r1:32B"    
# aide data_dir="example_tasks/house_prices" \
#     goal="Predict the sales price for each house" \
#     eval="Use the RMSE metric between the logarithm of the predicted and observed values."\
#     exec.timeout=120 \
#     agent.code.model=$LLM
python run_agent.py --competition-set experiments/splits/chosen_competions.txt --agent-id aide --n-workers 5 --n-seeds 3
# Capture the exit code of the aide command
EXIT_CODE=$? 

# Check if the aide command completed successfully (exit code 0)
if [ $EXIT_CODE -eq  6]; then
  echo "Aide command completed successfully. Copying outputs to Aichor path..."

    # --- DEBUGGING: Check if variables and paths exist ---
  # echo "AICHOR_OUTPUT_PATH is set to: $AICHOR_OUTPUT_PATH"

  echo "Checking if source ./logs directory exists..."
  ls -ld ./logs

  echo "Checking if source ./workspaces directory exists..."
  ls -ld ./workspaces

  # Copy the contents of the logs abd workspaces directory 
  # echo "Copying ./logs/* & ./workspaces/*  to $AICHOR_OUTPUT_PATH/"
  .aide-ds/bin/python upload_results.py
  
  # --- DEBUGGING: List contents of output path AFTER copy ---
  echo "Contents of AICHOR_OUTPUT_PATH after copy:"
  # ls -lR "$AICHOR_OUTPUT_PATH"

  echo "Outputs copied to specified local path for Aichor upload."
else
  echo "Aide command failed with exit code $EXIT_CODE. Skipping output copy."
  exit $EXIT_CODE # Ensure the Aichor job also fails
fi
# --- END: New lines to add --- 

echo "run.sh finished."
exit 0


# 1. Spooky author comp
# aide data_dir="data/aerial-cactus-identification" \
#       goal="Create a classifier capable of predicting whether a 32x32 aerial image contains a cactus" \
#       eval="Area Under the ROC Curve (AUC)" \
#       agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#       agent.steps=200 \
#       agent.code.max_new_tokens=2048 \
#       agent.code.temp=0.6 \
#       wandb.project="aerial-cactus-7b" \
#       exp_name="aerial-cactus_7b"


# 1. Spooky author comp
# aide data_dir="data/$competition_id/" \ 
#       goal="Predict the author of a sentence as one of Poe, Lovecraft, or Shelley" \
#       eval="Use multi-class logarithmic loss between predicted author probabilities and the true label." \
#       agent.code.model=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
#       agent.steps=200 \
#       agent.code.max_new_tokens=2048 \
#       agent.code.temp=0.6 \
#       wandb.project="spooky-author-14B" \
#       exp_name="14b_spooky_author"

 
# 2. aerial-cactus-identification
# aide data_dir="data/aerial-cactus-identification" \ 
#      goal="Create a classifier capable of predicting whether a 32x32 aerial image contains a cactus" \
#      eval="Area Under the ROC Curve (AUC)" \
#      agent.code.model= "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#      agent.steps=200 \
#      agent.code.max_new_tokens=2048 \
#      agent.code.temp=0.6 \
#      wandb.project="aerial-cactus-7b" \
#      exp_name="aerial-cactus_7b"

# 3. Random-Acts-0f-Pizza
# aide data_dir="data/$competition_id/" \ 
#      goal="Predict the probability that a textual request for pizza on Reddit resulted in the requester receiving pizza, probability that a textual request for pizza posted on Reddit will be successful" \
#      eval="Area Under the ROC Curve using the request text and associated metadata" \
#      agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
#      agent.steps=200 \
#      agent.code.temp=0.6 \
#      wandb.project="random-acts-of-pizza" \
#      exp_name="random-acts-of-pizza_DS_32B"


# 4. nomad2018-predict-transparent-conductors

# aide data_dir="data/nomad2018-predict-transparent-conductors/" \ 
#      goal="Predict formation energy (formation_energy_ev_natom) and bandgap energy (bandgap_energy_ev) for materials given their composition and structural properties" \
#      eval="Mean of column-wise Root Mean Squared Logarithmic Error (RMSLE) across the two target columns" \
#      agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#      agent.steps=200 \
#      agent.code.max_new_tokens=2048 \
#      agent.code.temp=0.6 \
#      wandb.project="aide-nomad2018" \
#      exp_name="7b_nomad2018"

## jigsaw-toxic-comment-classification-challenge

# aide data_dir="data/jigsaw-toxic-comment-classification-challenge"  \
#      goal="Predict the probability for each of six types of toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate) for each given comment text" \
#      eval="Mean column-wise ROC AUC" \
#      agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
#      agent.steps=200 \
#      agent.code.max_new_tokens=2048 \
#      agent.code.temp=0.6 \
#      wandb.project="aide-jigsaw-comment" \
#      exp_name="7b_jigsaw"


# zip -r Deepseek-r1-32b_nomad2018-logs.zip ./logs/32b_nomad2018 &
# zip -r Deepseek-r1-32b_nomad2018-workspace.zip ./32b_nomad2018 &


# python run_agent.py --competition-set experiments/splits/chosen_competions.txt --agent-id aide 