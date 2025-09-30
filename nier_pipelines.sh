#!/bin/bash
RESET="\e[0m"
RED="\e[31m"
GREEN="\e[32m"
YELLOW="\e[33m"
BLUE="\e[34m"
CYAN="\e[36m"
BOLD="\e[1m"


# Load Conda environment manually
source /home/standard/anaconda3/etc/profile.d/conda.sh

# Ensure Conda binary is in PATH
export PATH=/home/standard/anaconda3/bin:$PATH

# Log and PID Directory
LOG_DIR="$HOME/logs/NIER_SERVERS"
mkdir -p $LOG_DIR

PID_DIR="$HOME/logs/NIER_SERVERS/pids"
mkdir -p $PID_DIR

# Function to Check Conda Environment
check_conda_env() {
    EXPECTED_ENV=$1
    CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
    if [ "$CURRENT_ENV" != "$EXPECTED_ENV" ]; then
        echo -e "${RED}[ERROR] Expected Conda environment '$EXPECTED_ENV', but got '$CURRENT_ENV'.${RESET}"
        return 1
    else
        echo -e "${GREEN}[INFO] Conda environment '$EXPECTED_ENV' is correctly activated.${RESET}"
        return 0
    fi
}

# Start ChromaDB Server
echo -e "${BLUE}[INFO] Starting ChromaDB server...${RESET}"
conda activate chromadb
if [ $? -eq 0 ]; then
    if check_conda_env "chromadb"; then
        chroma run --path /home/0_code/RAG/ChromaDB/ts2vec_embedding_function \
                   --path /home/0_code/RAG/ChromaDB/ts2vec_embedding_function/NIER_DB > $LOG_DIR/chromadb.log 2>&1 &
        echo $! > $PID_DIR/chromadb.pid
        echo -e "${GREEN}[INFO] ChromaDB server is running in the background.${RESET}"
        echo -e "${CYAN}[INFO] PID: $(cat $PID_DIR/chromadb.pid), Log: $LOG_DIR/chromadb.log${RESET}"
    else
        echo -e "${RED}[ERROR] ChromaDB environment verification failed.${RESET}"
        exit 1
    fi
else
    echo -e "${RED}[ERROR] Failed to activate 'chromadb' Conda environment.${RESET}"
    exit 1
fi

# /home/0_code/NIER_Pipelines
# Start Pipelines Server
echo -e "${BLUE}[INFO] Starting Pipelines server...${RESET}"
conda activate pipelineDTW
if [ $? -eq 0 ]; then
    if check_conda_env "pipelineDTW"; then
        cd /home/0_code/NIER_Pipelines/ || {
            echo -e "${RED}[ERROR] Failed to change directory to /home/0_code/NIER_Pipelines/${RESET}"
            exit 1
        }
        bash start.sh > $LOG_DIR/pipelines.log 2>&1 &
        echo $! > $PID_DIR/pipelines.pid
        echo -e "${GREEN}[INFO] Pipelines server is running in the background.${RESET}"
        echo -e "${CYAN}[INFO] PID: $(cat $PID_DIR/pipelines.pid), Log: $LOG_DIR/pipelines.log${RESET}"
    else
        echo -e "${RED}[ERROR] Pipelines environment verification failed.${RESET}"
        exit 1
    fi
else
    echo -e "${RED}[ERROR] Failed to activate 'pipelineDTW' Conda environment.${RESET}"
    exit 1
fi


# Start OpenWebUI Server
echo -e "${BLUE}[INFO] Starting OpenWebUI server with extended timeout settings...${RESET}"
conda activate ollama
if [ $? -eq 0 ]; then
    if check_conda_env "ollama"; then
        export WEBUI_TIMEOUT=120
        export REQUEST_TIMEOUT=120
        export WEBUI_LOG_LEVEL=DEBUG
        echo -e "${CYAN}[INFO] Timeout settings: WEBUI_TIMEOUT=120, REQUEST_TIMEOUT=120${RESET}"
        open-webui serve > $LOG_DIR/openwebui.log 2>&1 &
        echo $! > $PID_DIR/openwebui.pid
        echo -e "${GREEN}[INFO] OpenWebUI server is running in the background.${RESET}"
        echo -e "${CYAN}[INFO] PID: $(cat $PID_DIR/openwebui.pid), Log: $LOG_DIR/openwebui.log${RESET}"
    else
        echo -e "${RED}[ERROR] OpenWebUI environment verification failed.${RESET}"
        exit 1
    fi
else
    echo -e "${RED}[ERROR] Failed to activate 'ollama' Conda environment.${RESET}"
    exit 1
fi

# All Servers Started
echo -e "${GREEN}[INFO] All servers have been successfully started!${RESET}"
echo -e "${CYAN}[INFO] Log Files:${RESET}"
echo -e "  - ${CYAN}Pipelines:${RESET} $LOG_DIR/pipelines.log"
echo -e "  - ${CYAN}ChromaDB:${RESET} $LOG_DIR/chromadb.log"
echo -e "  - ${CYAN}OpenWebUI:${RESET} $LOG_DIR/openwebui.log"


# Check if Servers are Running
echo -e "${BLUE}[INFO] Checking server statuses...${RESET}"
for service in chromadb pipelines openwebui; do
    if [ -f "$PID_DIR/$service.pid" ]; then
        PID=$(cat $PID_DIR/$service.pid)
        if ps -p $PID > /dev/null; then
            echo -e "${GREEN}[INFO] $service server (PID: $PID) is running.${RESET}"
        else
            echo -e "${RED}[ERROR] $service server (PID: $PID) is not running.${RESET}"
            rm -f $PID_DIR/$service.pid
        fi
    else
        echo -e "${YELLOW}[WARNING] No PID file found for $service server.${RESET}"
    fi
done

# Display All Logs in Real-Time
echo -e "${BLUE}[INFO] Displaying all server logs in real-time...${RESET}"
tail -f $LOG_DIR/pipelines.log \
       $LOG_DIR/chromadb.log \
       $LOG_DIR/openwebui.log