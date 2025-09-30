#!/bin/bash

# NIER Pipeline 완전 종료 스크립트
# 작성일: 2025-01-04
# 용도: 모든 NIER 관련 프로세스를 안전하게 종료

echo "============================================"
echo "   NIER Pipeline 서비스 종료 스크립트"
echo "============================================"
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 종료할 프로세스 목록
declare -a PROCESSES=(
    "nier_pipelines.sh"
    "open-webui"
    "chroma"
    "uvicorn.*12001"  # Pipeline API
    "ollama_llama_server"
    "tail.*NIER_SERVERS"  # 로그 tail 프로세스
)

# 1. NIER Pipeline 스크립트 종료
echo -e "${YELLOW}[1/5] NIER Pipeline 스크립트 종료 중...${NC}"
pkill -f "nier_pipelines.sh" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ NIER Pipeline 스크립트 종료됨${NC}"
else
    echo -e "${RED}✗ NIER Pipeline 스크립트가 실행 중이 아님${NC}"
fi

# 2. OpenWebUI 종료
echo -e "${YELLOW}[2/5] OpenWebUI 종료 중...${NC}"
pkill -f "open-webui serve" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ OpenWebUI 종료됨${NC}"
else
    echo -e "${RED}✗ OpenWebUI가 실행 중이 아님${NC}"
fi

# 3. Pipeline API (uvicorn) 종료
echo -e "${YELLOW}[3/5] Pipeline API 종료 중...${NC}"
pkill -f "uvicorn.*port 12001" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Pipeline API 종료됨${NC}"
else
    echo -e "${RED}✗ Pipeline API가 실행 중이 아님${NC}"
fi

# 4. ChromaDB 종료
echo -e "${YELLOW}[4/5] ChromaDB 종료 중...${NC}"
pkill -f "chromadb" 2>/dev/null
pkill -f "chroma run" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ChromaDB 종료됨${NC}"
else
    echo -e "${RED}✗ ChromaDB가 실행 중이 아님${NC}"
fi

# 5. 로그 tail 프로세스 종료
echo -e "${YELLOW}[5/5] 로그 모니터링 프로세스 종료 중...${NC}"
pkill -f "tail.*NIER_SERVERS" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 로그 모니터링 종료됨${NC}"
else
    echo -e "${RED}✗ 로그 모니터링이 실행 중이 아님${NC}"
fi

echo ""
echo "============================================"
echo "   남은 프로세스 확인"
echo "============================================"

# 남은 프로세스 확인
REMAINING=$(ps aux | grep -E "(nier|open-webui|chroma|uvicorn.*12001)" | grep -v grep | grep -v "stop_all_nier")

if [ -z "$REMAINING" ]; then
    echo -e "${GREEN}✓ 모든 NIER 관련 프로세스가 성공적으로 종료되었습니다.${NC}"
else
    echo -e "${YELLOW}⚠ 다음 프로세스가 아직 실행 중입니다:${NC}"
    echo "$REMAINING"
    echo ""
    echo -e "${YELLOW}강제 종료하시겠습니까? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "${RED}강제 종료 진행 중...${NC}"
        ps aux | grep -E "(nier|open-webui|chroma|uvicorn.*12001)" | grep -v grep | grep -v "stop_all_nier" | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        echo -e "${GREEN}✓ 강제 종료 완료${NC}"
    fi
fi

echo ""
echo "============================================"
echo "   포트 상태 확인"
echo "============================================"

# 포트 확인
echo "사용 중인 포트:"
netstat -tlnp 2>/dev/null | grep -E "(8080|8000|12001)" | awk '{print $4 " -> " $7}' || echo "포트 확인에 root 권한이 필요합니다."

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   NIER Pipeline 종료 완료${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "다시 시작하려면 다음 명령어를 실행하세요:"
echo "  cd /home/0_code/NIER_Pipelines"
echo "  ./nier_pipelines.sh"
echo ""