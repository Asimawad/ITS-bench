#!/bin/bash
set -x # Print commands and their arguments as they are executed
pwd
cd ../../../
echo $(pwd)
pwd
source .venv/bin/activate
cd ${HOME_DIR}


# determine hardware available
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE
# check that we can use the GPU in PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"
# check that we can use the GPU in TensorFlow
python -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))"

# convert $TIME_LIMIT_SECS to more readable format for prompt
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# overwrite instructions.txt with instructions_obfuscated.txt if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /${HOME_DIR}/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/instructions_obfuscated.txt /home/instructions.txt
fi
echo "________________"
ls -a
echo "________________"
# start a new file to store the full instructions, starting with general instructions
cp ${AGENT_DIR}/instructions.txt ${AGENT_DIR}/full_instructions.txt
cat ${AGENT_DIR}/full_instructions.txt


sed -i 's|/'${HOME_DIR}'/||g' ${AGENT_DIR}/full_instructions.txt


# substitute env variables into additional_notes.txt and append result to full_instructions.txt
echo "" >> ${AGENT_DIR}/full_instructions.txt
envsubst < ${AGENT_DIR}/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt
# finally, append the comp instructions, with a linebreak in between
printf "\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt

# overwrite description.md with description_obfuscated.md if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /${HOME_DIR}/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv ${HOME_DIR}/data/description_obfuscated.md ${HOME_DIR}/data/description.md
fi
cat ${HOME_DIR}/data/description.md >> ${AGENT_DIR}/full_instructions.txt


timeout $TIME_LIMIT_SECS
aide \
  data_dir="${HOME_DIR}/data/" \
  desc_file="${AGENT_DIR}/full_instructions.txt" \
  agent.code.model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  "$@" # forward the bash arguments to aide
if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi
