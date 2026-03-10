#!/bin/bash
# ============================================================
# Quick Start Script for Video Translation Tool
# ============================================================
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  Video Translation Tool - Quick Start${NC}"
echo -e "${CYAN}================================================${NC}"

# Check dependencies
check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${YELLOW}Warning: $1 not found. Please install it.${NC}"
        return 1
    fi
    return 0
}

echo -e "\n${GREEN}Checking dependencies...${NC}"
check_cmd python3
check_cmd ffmpeg
check_cmd ffprobe

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

# Install requirements
echo -e "${GREEN}Installing Python dependencies...${NC}"
pip install -r requirements.txt -q

# Create directories
mkdir -p input output temp

echo -e "\n${CYAN}================================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""
echo -e "Usage:"
echo -e "  ${YELLOW}# Translate a local video${NC}"
echo -e "  python main.py translate input/video.mp4 -t zh"
echo ""
echo -e "  ${YELLOW}# Translate a YouTube video${NC}"
echo -e '  python main.py translate "https://youtube.com/watch?v=xxx" -t zh'
echo ""
echo -e "  ${YELLOW}# Translate a Bilibili video${NC}"
echo -e '  python main.py translate "https://bilibili.com/video/BVxxx" -t en'
echo ""
echo -e "  ${YELLOW}# Use specific backend${NC}"
echo -e "  python main.py translate video.mp4 -t ja -b ollama"
echo -e "  python main.py translate video.mp4 -t zh -b openai_api"
echo ""
echo -e "  ${YELLOW}# List options${NC}"
echo -e "  python main.py list-voices"
echo -e "  python main.py list-backends"
echo ""
echo -e "${CYAN}Config: Edit config.yaml to customize settings${NC}"
echo -e "${CYAN}Docker: docker compose run translate translate input/video.mp4 -t zh${NC}"
