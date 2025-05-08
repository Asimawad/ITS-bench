#!/bin/bash

# run_quiet.sh - Run the agent with minimal logging output
# Usage: ./run_quiet.sh [competition-set] [agent-id] [n-workers] [n-seeds]

# Set default values
COMPETITION_SET=${1:-"custom-test.txt"}
AGENT_ID=${2:-"aide"}
N_WORKERS=${3:-"2"}
N_SEEDS=${4:-"1"}

echo "Running with quiet mode enabled..."
echo "  Competition set: $COMPETITION_SET"
echo "  Agent ID: $AGENT_ID"
echo "  Workers: $N_WORKERS"
echo "  Seeds: $N_SEEDS"
echo ""
echo "Only warnings and important messages will be shown."
echo "All logs are still being saved to the run directory."
echo ""

# Run the custom agent script with quiet mode
python run_agent_custom.py \
  --agent-id "$AGENT_ID" \
  --competition-set "$COMPETITION_SET" \
  --n-workers "$N_WORKERS" \
  --n-seeds "$N_SEEDS" \
  --quiet

# Print completion message
echo ""
echo "Run completed. Check the 'runs' directory for full logs." 