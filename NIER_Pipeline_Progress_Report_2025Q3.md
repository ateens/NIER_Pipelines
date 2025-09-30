# NIER Pipeline 업무 진행 발표
## 2025년 3분기 개선 및 4분기 계획

**발표일**: 2025년 9월 30일  
**발표자**: ateens  
**프로젝트**: NIER RAG Pipeline 성능 개선 및 아키텍처 현대화

---

## 📊 Executive Summary

### 완료된 작업 (2025년 9월)
1. ✅ **시계열 인코더 교체**: TS2Vec → T-Rep (성능 향상 확인)
2. ✅ **LLM 모델 교체**: Bllossom-8B → qwen3:30b (30B 파라미터)
3. ✅ **멀티턴 대화 구현**: 컨텍스트 유지 대화 시스템 구축

### 성과
- **이상 탐지 정확도 향상**: Class=3 Recall 13.7% → 71.8% (5.2배)
- **전체 정확도 향상**: 64.8% → 89.0%
- **임베딩 효율성**: 차원 320 → 128 (60% 감소), 저장 공간 75% 절감
- **사용자 경험 개선**: 멀티턴 대화 지원 (최근 5턴 컨텍스트 유지)

### 향후 계획 (~2025년 11월)
- 아키텍처 현대화 (vLLM, Qdrant, Docker)
- 시스템 확장성 및 안정성 강화

---

## 🎯 Part 1: 완료된 개선 작업

### 1. 시계열 인코더 교체 (TS2Vec → T-Rep)

#### 📌 교체 배경
- **기존 문제**: TS2Vec의 낮은 이상 탐지 성능
  - class=3 (이상) 데이터 탐지 실패 (Recall: 0.0%)
  - 정상/이상 구분 불가
  
- **선택 근거**: T-Rep의 향상된 표현 학습
  - 시간 임베딩 활용 (t2v_sin)
  - 계층적 대조 학습
  - 성분별 특화 모델 (SO2, O3, CO, NO, NO2)

#### 🔧 구현 내용

**1) 성분별 T-Rep 모델 적용**
```python
# 각 오염 물질별로 독립적인 T-Rep 모델 사용
self.embedding_model_path = {
    "SO2": "/home/0_code/RAG/T-Rep/training/training/v0_1_1__NIER_SO2_20250422_210513/model.pt",
    "O3": "/home/0_code/RAG/T-Rep/training/training/v1_0_0__NIER_O3_20250410_053440/model.pt",
    "CO": "/home/0_code/RAG/T-Rep/training/training/v0_1_1__NIER_CO_20250424_160647/model.pt",
    "NO": "/home/0_code/RAG/T-Rep/training/training/v0_1_1__NIER_NO_20250422_183519/model.pt",
    "NO2": "/home/0_code/RAG/T-Rep/training/training/v0_1_1__NIER_NO2_20250424_055403/model.pt"
}
```

**2) ChromaDB 컬렉션 재구성**
```python
# 기존: time_series_collection (320차원, TS2Vec)
# 신규: time_series_collection_trep (128차원, T-Rep)
```

**3) 기존 데이터 재임베딩**
- 스크립트: `scripts/reembed_with_trep_direct.py`
- 처리 데이터: 143,680개 시계열 레코드
- 소요 시간: 약 12시간

#### 📈 성능 비교 결과

**1) 이상 탐지 성능 (Classification Benchmark)**

| 지표 | TS2Vec | T-Rep | 개선 |
|------|--------|-------|------|
| **정확도 (Accuracy)** | 0.648 | 0.890 | +0.242 |
| **F1-Score** | 0.473 | 0.851 | +0.377 |
| **Class=3 Recall** | **0.137** | **0.718** | **+0.580** |

*Class=3 Recall: 베이스라인 이상(이상 데이터)를 얼마나 잘 찾아내는가*

**2) 시스템 효율성**

| 지표 | TS2Vec | T-Rep | 개선 |
|------|--------|-------|------|
| **임베딩 차원** | 320 | 128 | -60% |
| **저장 공간** (테스트셋) | 7.24 MB | 1.81 MB | -75% |
| **검색 속도** (k=10) | ~14 ms | ~14 ms | 동등 |

*임베딩 생성 속도는 원소별로 상이 (SO2: 24→22ms, NO2: 5→7ms)*

#### 🎯 핵심 성과
- **이상 탐지 능력 향상**: Class=3 Recall 13.7% → 71.8% (5.2배 개선)
- **전체 정확도 향상**: 64.8% → 89.0%
- **저장 효율성**: 75% 저장 공간 절약
- **임베딩 차원 감소**: 320차원 → 128차원

---

### 2. LLM 모델 교체 (Bllossom-8B → qwen3:30b)

#### 📌 교체 배경
- **성능 한계**: Bllossom-8B의 제한적인 추론 능력
- **멀티턴 필요성**: 연속 대화를 통한 UX 개선
- **모델 선택**: qwen3:30b (Qwen2.5 기반, 30B 파라미터)

#### 🔧 구현 내용

**1) Ollama 모델 설정**
```bash
# qwen3:30b 모델 다운로드 및 설정
ollama pull qwen3:30b
```

**2) Pipeline Valves 설정 변경**
```python
self.valves.TEXT_TO_TS_TO_EMBEDDING_MODEL = "qwen3:30b"
```

#### 📊 기대 효과
- **추론 능력 향상**: 8B → 30B 파라미터 (3.75배 증가)
- **복잡한 쿼리 처리**: 다단계 추론 및 맥락 이해 개선
- **한국어 지원**: 다국어 모델로 영어/한국어 프롬프팅 가능

---

### 3. 멀티턴 대화 구현

#### 📌 구현 배경
- **기존 시스템**: 단발성 질의응답 (컨텍스트 미유지)
- **사용자 요구**: 자연스러운 연속 대화
  ```
  User: "632132 SO2 데이터 보여줘"
  Bot: [분석 결과]
  User: "이게 정상이야?" ← 이전 대화 참조 불가
  ```

#### 🔧 구현 내용

**1) 대화 히스토리 관리**
```python
def explain_query(self, ..., conversation_history: List[dict] = None):
    """시계열 분석 시 이전 대화 맥락 포함"""
    messages = [
        {"role": "system", "content": "...이전 대화 맥락을 고려하여..."},
        *recent_messages[-10:],  # 최근 10개 메시지 (5턴)
        {"role": "user", "content": current_prompt}
    ]
    return call_ollama_chat_api(messages=messages)
```

**2) 통합 멀티턴 처리**
```python
def handle_general_question_with_history(self, question, conversation_history):
    """일반 질문도 대화 히스토리 포함"""
    messages = [system_prompt] + recent_messages[-10:] + [current_question]
    return call_ollama_chat_api(messages=messages)
```

#### 🎯 핵심 기능

**1) 지시어 이해**
```
User: "632132 SO2 24-01-01~01-10 조회"
Bot: "정상입니다. 연관 측정소와 유사도..."

User: "이게 정상이야?" ← "이게" = 632132 SO2 데이터
Bot: "네, 앞서 분석한 632132 SO2는 정상입니다..."

User: "베이스라인 이상 확률은?" ← 이전 대화 맥락 유지
Bot: "632132 지역의 베이스라인 이상 확률은 12.5%..."
```

**2) 크로스 쿼리 타입 대화**
```
User: "111123 NO2 22-08-03~09-10 조회" [시계열 쿼리]
Bot: [시계열 분석 결과]

User: "SO2는 뭐야?" [일반 질문]
Bot: "SO2는 이산화황으로..."

User: "아까 분석한 지역과 관련 있어?" [맥락 참조]
Bot: "111123 지역의 NO2와 SO2는..." ← 이전 시계열 분석 기억
```

**3) 토큰 효율성**
- 최근 10개 메시지만 유지 (약 5턴)
- 긴 대화에서도 메모리 안전
- qwen3:30b의 32K 컨텍스트 윈도우 활용

#### 📊 구현 방식 비교

| 항목 | 이전 | 현재 |
|------|------|------|
| **대화 방식** | 단발 질의응답 | 연속 대화 |
| **컨텍스트** | 없음 | 최근 5턴 유지 |
| **지시어 이해** | 불가 | 가능 ("이게", "그거", "아까") |
| **크로스 참조** | 불가 | 가능 (시계열↔일반) |
| **API 형식** | generate | chat |

---

## 🚀 Part 2: 향후 계획 (2025년 4분기)

### 마이그레이션 로드맵 (~2025년 11월)

#### Phase 1: LLM 인프라 개선

**1.1 Ollama → vLLM 마이그레이션**

현재 문제점:
- Ollama: 단일 인스턴스, 제한적 동시성
- 로드밸런싱 부재
- 모니터링 도구 부족

vLLM 도입 이유:
- ✅ PagedAttention으로 메모리 효율성 극대화
- ✅ Continuous batching으로 처리량 증가
- ✅ 여러 모델 동시 서빙 가능
- ✅ OpenAI 호환 API
- ✅ 로드밸런싱 내장

구현 계획:
```bash
# vLLM 서버 설정
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-30B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

기대 효과:
- 처리량: 2-3배 증가
- 동시 요청: 10+ 사용자
- GPU 활용률: 90%+

**1.2 로드밸런싱 구축**

아키텍처:
```
                    ┌─────────────┐
                    │  Nginx LB   │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ vLLM #1   │   │ vLLM #2   │   │ vLLM #3   │
    │ GPU 0     │   │ GPU 1     │   │ GPU 2     │
    └───────────┘   └───────────┘   └───────────┘
```

구현 도구:
- Nginx (라운드 로빈 + 헬스 체크)
- Redis (세션 유지)
- Prometheus + Grafana (모니터링)

---

#### Phase 2: 벡터 DB 개선

**2.1 ChromaDB → Qdrant 마이그레이션**

현재 문제점:
- ChromaDB: 제한적 스케일링
- 고급 필터링 부족
- 분산 처리 미지원

Qdrant 도입 이유:
- ✅ 대규모 데이터 처리 (억 단위)
- ✅ 샤딩/복제 지원
- ✅ 고급 필터링 (날짜, 범위, geo 등)
- ✅ gRPC 지원 (빠른 통신)
- ✅ 클라우드 네이티브

구현 계획:
```python
# Qdrant 컬렉션 생성
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)
client.create_collection(
    collection_name="nier_timeseries_trep",
    vectors_config=VectorParams(size=128, distance=Distance.COSINE),
    optimizers_config={
        "indexing_threshold": 20000,
        "memmap_threshold": 50000
    }
)
```

마이그레이션 전략:
1. Qdrant 병렬 운영 (기존 ChromaDB 유지)
2. 스크립트로 데이터 복사
3. 검증 후 ChromaDB 단계적 폐기

기대 효과:
- 검색 속도: 2-5배 향상
- 동시 쿼리: 100+ 지원
- 필터링 유연성 증가

---

#### Phase 3: 오케스트레이션 개선

**3.1 OpenWebUI Pipelines → Langflow/LangGraph**

현재 문제점:
- OpenWebUI Pipelines: 단순 스크립트 기반
- 복잡한 플로우 관리 어려움
- 디버깅 및 모니터링 제한적

Langflow vs LangGraph 비교:

| 항목 | Langflow | LangGraph |
|------|----------|-----------|
| **UI** | 드래그앤드롭 GUI | 코드 기반 |
| **학습 곡선** | 낮음 | 중간 |
| **유연성** | 중간 | 높음 |
| **디버깅** | 시각적 | 코드 레벨 |
| **배포** | REST API | Python 통합 |
| **적합성** | 빠른 프로토타입 | 복잡한 로직 |

권장: LangGraph 우선, Langflow 보조

LangGraph 구현 예시:
```python
from langgraph.graph import Graph

# NIER Pipeline을 그래프로 표현
graph = Graph()

# 노드 정의
graph.add_node("parse_query", parse_query_node)
graph.add_node("validate", validate_node)
graph.add_node("fetch_data", fetch_data_node)
graph.add_node("compare_stations", compare_stations_node)
graph.add_node("embed", embed_node)
graph.add_node("search_vector", search_vector_node)
graph.add_node("explain_llm", explain_llm_node)

# 엣지 정의 (조건부 라우팅)
graph.add_conditional_edges(
    "parse_query",
    route_query_type,
    {
        "time_series": "validate",
        "general": "explain_llm"
    }
)

# 실행
result = graph.run(user_input)
```

기대 효과:
- 복잡한 플로우 관리 용이
- A/B 테스트 및 실험 간소화
- 에러 처리 및 재시도 로직 강화

---

#### Phase 4: 컨테이너화 (Docker)

**4.1 우선순위 컨테이너화**

Phase 4-1: 기본 서비스 (1차)
```yaml
# docker-compose.yml
version: '3.8'
services:
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE

  openwebui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    volumes:
      - ./openwebui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ./ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Phase 4-2: Pipeline 컨테이너화 (2차)
```dockerfile
# Dockerfile.pipeline
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipelines/ ./pipelines/
COPY NIERModules/ ./NIERModules/

CMD ["uvicorn", "pipelines.main:app", "--host", "0.0.0.0", "--port", "12001"]
```

**4.2 마이그레이션 전략**

단계적 접근:
1. Week 1-2: ChromaDB 도커화 + 테스트
2. Week 3-4: OpenWebUI 도커화 + 통합
3. Week 5-6: Ollama 도커화 + GPU 설정
4. Week 7-8: Pipeline 도커화
5. Week 9-10: 전체 시스템 통합 테스트

롤백 계획:
- 각 단계마다 기존 시스템 병렬 운영
- 문제 발생 시 즉시 기존 시스템으로 복귀
- 데이터 백업 자동화

**4.3 기대 효과**

운영 측면:
- ✅ 환경 일관성: 개발/스테이징/프로덕션 동일
- ✅ 배포 자동화: CI/CD 파이프라인 구축 가능
- ✅ 리소스 격리: 서비스별 독립적 관리
- ✅ 빠른 복구: 컨테이너 재시작만으로 문제 해결

개발 측면:
- ✅ 로컬 개발 환경: docker-compose up으로 전체 스택 실행
- ✅ 의존성 관리: Conda 환경 충돌 없음
- ✅ 버전 관리: 이미지 태그로 버전 추적
- ✅ 팀 협업: 동일 환경에서 작업

---

## 📊 전체 타임라인

### 2025년 9월 (완료)
- ✅ T-Rep 인코더 통합
- ✅ qwen3:30b 모델 교체
- ✅ 멀티턴 대화 구현
- ✅ 성능 벤치마크

### 2025년 10월 (예정)
- 🔄 vLLM 마이그레이션
- 🔄 로드밸런싱 구축
- 🔄 Qdrant POC

### 2025년 11월 (예정)
- 🔄 Qdrant 본격 전환
- 🔄 LangGraph 도입
- 🔄 Docker 컨테이너화 (Phase 1-3)

### 2025년 12월 (버퍼)
- 🔄 통합 테스트
- 🔄 모니터링 시스템 구축
- 🔄 문서화 및 운영 가이드

---

## 💡 핵심 성과 요약

### 단기 성과 (9월)
1. **이상 탐지 성능 향상**
   - Class=3 Recall: 13.7% → 71.8% (5.2배)
   - 전체 정확도: 64.8% → 89.0%
   - F1-Score: 0.473 → 0.851

2. **시스템 효율성 개선**
   - 임베딩 차원: 320 → 128 (60% 감소)
   - 저장 공간: 75% 절감
   - 검색 속도: 유지 (~14ms)

3. **사용자 경험 개선**
   - 멀티턴 대화 지원 (최근 5턴 컨텍스트 유지)
   - LLM 모델 업그레이드 (8B → 30B)

### 중장기 목표 (10-12월)
1. **확장성**: vLLM + 로드밸런싱으로 동시 사용자 10+
2. **성능**: Qdrant로 검색 속도 2-5배 향상
3. **유지보수성**: Docker로 배포 자동화 및 환경 일관성
4. **유연성**: LangGraph로 복잡한 플로우 관리

---

## 🔧 기술 스택 변화

### Before (2025년 8월)
```
┌─────────────────────────────────────┐
│  Frontend: OpenWebUI                │
├─────────────────────────────────────┤
│  Pipeline: Custom Python Scripts    │
│  (Conda Environment)                │
├─────────────────────────────────────┤
│  LLM: Ollama (Bllossom-8B)          │
│  Encoder: TS2Vec (320d)             │
│  Vector DB: ChromaDB                │
├─────────────────────────────────────┤
│  DB: PostgreSQL                     │
└─────────────────────────────────────┘
```

### After (2025년 12월 예정)
```
┌─────────────────────────────────────┐
│  Frontend: OpenWebUI (Dockerized)   │
├─────────────────────────────────────┤
│  Orchestration: LangGraph           │
│  (Dockerized Pipeline)              │
├─────────────────────────────────────┤
│  LLM: vLLM (qwen3:30b)              │
│  Load Balancer: Nginx               │
│  Encoder: T-Rep (128d)              │
│  Vector DB: Qdrant (Sharded)        │
├─────────────────────────────────────┤
│  DB: PostgreSQL                     │
│  Monitoring: Prometheus + Grafana   │
└─────────────────────────────────────┘
```

---

## 📈 정량적 개선 지표

### 성능 지표
| 항목 | Before | After | 개선율 |
|------|--------|-------|--------|
| Class=3 Recall | 13.7% | 71.8% | +58.1% |
| 전체 정확도 | 64.8% | 89.0% | +24.2% |
| F1-Score | 0.473 | 0.851 | +37.8% |
| 임베딩 차원 | 320 | 128 | -60% |
| 저장 공간 (테스트셋) | 7.24 MB | 1.81 MB | -75% |
| 검색 속도 (k=10) | ~14 ms | ~14 ms | 유사 |
| 대화 컨텍스트 | 없음 | 5턴 | 신규 |
| LLM 파라미터 | 8B | 30B | +275% |

### 예상 개선 (4분기)
| 항목 | Current | Target | 방법 |
|------|---------|--------|------|
| 동시 사용자 | 3-5명 | 10-20명 | vLLM + LB |
| 검색 속도 | 50ms | 10-20ms | Qdrant |
| 배포 시간 | 30분 | 5분 | Docker |
| GPU 활용률 | 60% | 90% | vLLM |

---

## 🎯 리스크 및 대응 방안

### 기술적 리스크

**1. vLLM 마이그레이션 실패**
- 리스크: 모델 호환성 이슈, 메모리 부족
- 대응: Ollama 병렬 운영, 단계적 전환, 충분한 테스트

**2. Qdrant 성능 미달**
- 리스크: 예상보다 느린 검색 속도
- 대응: ChromaDB 유지, 하이브리드 운영, 인덱스 최적화

**3. Docker 리소스 오버헤드**
- 리스크: 컨테이너 오버헤드로 성능 저하
- 대응: 리소스 제한 조정, 호스트 네트워크 모드, 프로파일링

### 일정 리스크

**4. 마이그레이션 지연**
- 리스크: 예상보다 긴 작업 시간
- 대응: 우선순위 재조정, 단계별 완료 기준, 버퍼 시간 확보

### 운영 리스크

**5. 서비스 중단**
- 리스크: 마이그레이션 중 서비스 불가
- 대응: 블루-그린 배포, 롤백 계획, 공지 및 유지보수 시간 확보

---

## ✨ 결론

### 완료된 성과
2025년 9월 한 달간 **NIER Pipeline의 핵심 기능을 개선**했습니다:
- 이상 탐지 성능 향상 (Class=3 Recall: 13.7% → 71.8%)
- 전체 정확도 향상 (64.8% → 89.0%)
- LLM 모델 업그레이드 (Bllossom-8B → qwen3:30b)
- 멀티턴 대화 지원 추가

### 향후 방향
2025년 4분기에는 **시스템의 확장성과 안정성**에 집중합니다:
- vLLM + 로드밸런싱으로 동시 사용자 처리 능력 향상
- Qdrant로 대규모 데이터 처리 준비
- Docker로 배포 자동화 및 운영 효율화

### 기대 효과
이러한 개선을 통해 NIER Pipeline은:
- **더 정확한**: 향상된 이상 탐지
- **더 빠른**: 최적화된 인프라
- **더 편리한**: 자연스러운 대화형 인터페이스
- **더 안정적인**: 컨테이너 기반 운영

시스템으로 발전할 것입니다.

---

**문의 및 피드백**: ateens (ateens8120@gmail.com)  
**프로젝트 저장소**: [github.com/ateens/NIER_Pipelines](https://github.com/ateens/NIER_Pipelines)  
**마지막 업데이트**: 2025년 9월 30일
