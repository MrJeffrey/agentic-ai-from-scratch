#!/bin/bash

# Define colors for status messages
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Python version - use 3.13 for compatibility with pinned dependencies
PYTHON=${PYTHON:-python3.13}

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   Tutorial 1: The Interaction Loop     ${NC}"
echo -e "${BLUE}=========================================${NC}"

# 1. Check for Python
if ! command -v $PYTHON &> /dev/null; then
    echo -e "${RED}Error: Python3 is not installed.${NC}"
    echo -e "Please install Python 3.10+ from https://python.org"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}Found Python ${PYTHON_VERSION}${NC}"

# 2. Check/Create Virtual Environment
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    $PYTHON -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo -e "${GREEN}Virtual environment found.${NC}"
fi

# 3. Activate Virtual Environment
source venv/bin/activate

# 4. Install Dependencies
if [ -f "requirements.txt" ]; then
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}Dependencies installed.${NC}"
else
    echo -e "${RED}Warning: requirements.txt not found!${NC}"
fi

# Function to setup API key interactively
setup_api_key() {
    echo ""
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}   First-Time Setup                      ${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
    echo -e "This tutorial requires an ${YELLOW}OpenAI API key${NC} to work."
    echo ""
    echo -e "${GREEN}What is an API key?${NC}"
    echo "  An API key is like a password that lets this tutorial"
    echo "  communicate with OpenAI's AI models (GPT-4, etc.)."
    echo ""
    echo -e "${GREEN}How to get your OpenAI API key:${NC}"
    echo "  1. Go to: https://platform.openai.com/api-keys"
    echo "  2. Sign up or log in to your OpenAI account"
    echo "  3. Click '+ Create new secret key'"
    echo "  4. Copy the key (starts with 'sk-')"
    echo ""
    echo -e "${YELLOW}Note: OpenAI charges for API usage (typically a few cents per tutorial).${NC}"
    echo -e "${YELLOW}New accounts usually get free credits to start.${NC}"
    echo ""
    echo "========================================="
    echo ""

    # Prompt for API key
    while true; do
        echo -e "Paste your OpenAI API key here (or type 'quit' to exit):"
        read -r api_key

        if [ "$api_key" = "quit" ] || [ "$api_key" = "q" ]; then
            echo -e "${YELLOW}Setup cancelled. Run this script again when you have your API key.${NC}"
            exit 0
        fi

        # Validate key format (starts with sk- and is reasonably long)
        if [[ "$api_key" == sk-* ]] && [ ${#api_key} -gt 20 ]; then
            break
        else
            echo ""
            echo -e "${RED}That doesn't look like a valid OpenAI API key.${NC}"
            echo "API keys start with 'sk-' and are about 50+ characters long."
            echo ""
        fi
    done

    # Create .env file in parent directory (shared across all tutorials)
    echo "OPENAI_API_KEY=$api_key" > ../.env
    echo ""
    echo -e "${GREEN}API key saved! This key will work for all tutorials.${NC}"
    echo -e "${YELLOW}Tip: To change your API key later, edit the .env file in the Tutorials folder.${NC}"
    echo ""
}

# 5. Check for .env file in parent directory (shared across all tutorials)
ENV_FILE="../.env"
needs_setup=false

if [ ! -f "$ENV_FILE" ]; then
    needs_setup=true
else
    source "$ENV_FILE"
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
        needs_setup=true
    fi
fi

if [ "$needs_setup" = true ]; then
    setup_api_key
    source "$ENV_FILE"
fi

# 6. Run the tutorial
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   Launching Tutorial...                 ${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

$PYTHON chat_loop.py

echo ""
echo -e "${GREEN}Tutorial complete!${NC}"
