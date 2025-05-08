#!/bin/bash
# Simple script to test OpenAI API models with proper environment setup

set -e

# Set colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== OpenAI Model Connection Test =====${NC}"
echo "This script will test connectivity to different LLM models"

# Activate the virtual environment if available
if [ -d ".aide-ds" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .aide-ds/bin/activate
    echo -e "${GREEN}Virtual environment activated${NC}"
fi

# Check for required Python packages
if ! python -c "import openai" &> /dev/null; then
    echo -e "${RED}Error: openai package not found. Installing...${NC}"
    pip install openai
fi

if ! python -c "import dotenv" &> /dev/null; then
    echo -e "${YELLOW}Warning: python-dotenv package not found. Installing...${NC}"
    pip install python-dotenv
fi

# Make the test script executable
chmod +x scripts/test_models.py

# Check if API key is provided as argument or use environment variable
API_KEY=${1:-$OPENAI_API_KEY}
if [ -n "$API_KEY" ]; then
    echo -e "${GREEN}Using provided API key${NC}"
    python scripts/test_models.py --model all --api-key "$API_KEY"
else
    echo -e "${YELLOW}Using API key from environment variable (if set)${NC}"
    python scripts/test_models.py --model all
fi

echo -e "${BLUE}===== Test Complete =====${NC}" 