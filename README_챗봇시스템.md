# 🤖 수요 예측 챗봇 시스템

## 📋 전체 구조

```
CSV 업로드 → 분석/예측 → 시각화 → LLM 챗봇 해석
```

---

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install pandas numpy scikit-learn
pip install pymongo
pip install plotly kaleido  # 시각화
pip install openai  # 또는 anthropic
pip install flask flask-cors
```

### 2. MongoDB 실행

```bash
# MongoDB 시작
mongod

# 또는 Docker
docker run -d -p 27017:27017 mongo
```

### 3. 환경변수 설정

```bash
# LLM API 키 설정
export OPENAI_API_KEY="your-api-key"
# 또는
export ANTHROPIC_API_KEY="your-api-key"
```

### 4. 서버 실행

```bash
python chatbot_web_api.py
```

### 5. 프론트엔드 열기

```bash
# frontend_example.html을 브라우저에서 열기
# 또는
python -m http.server 8000
# http://localhost:8000/frontend_example.html
```

---

## 📁 파일 구조

```
blinkit-data-analysis/
├── prediction_service.py          # 수량/금액 예측 서비스
├── visualization_service.py       # 시각화 생성 서비스
├── llm_chatbot_service.py         # LLM 챗봇 서비스
├── chatbot_web_api.py             # Flask 웹 API
├── frontend_example.html          # 프론트엔드 예시
├── auto_feature_pipeline.py       # 자동화 파이프라인
├── mongodb_schema.py              # MongoDB 서비스
└── 챗봇_시스템_설계.md            # 상세 설계 문서
```

---

## 🔄 사용 흐름

### 1. CSV 업로드
```python
from mongodb_schema import MongoDBService

mongo = MongoDBService()
result = mongo.upload_csv(
    user_id="user123",
    file_path="data/blinkit_data.csv",
    file_name="blinkit_data.csv",
    file_size=1024000
)

file_id = result['file_id']
```

### 2. 분석 및 예측
```python
from prediction_service import PredictionService
from visualization_service import VisualizationService
from llm_chatbot_service import LLMChatbotService

# 예측
prediction_service = PredictionService()
predictions = prediction_service.predict_quantity_and_amount(file_id, forecast_days=7)

# 시각화
viz_service = VisualizationService()
charts = viz_service.create_forecast_charts(predictions, file_id)

# LLM 인사이트
llm_service = LLMChatbotService(api_key="your-api-key")
insights = llm_service.generate_insights(predictions, predictions['metrics'])
```

### 3. 챗봇 대화
```python
# 질문하기
answer = llm_service.answer_question(
    question="이번 주 수량 예측이 어떻게 되나요?",
    file_id=file_id,
    predictions=predictions
)
```

---

## 🌐 API 엔드포인트

### POST `/api/analyze-and-predict`
분석 및 예측 실행

**Request:**
```json
{
  "file_id": "file_001",
  "forecast_days": 7
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "quantity": {"dates": [...], "predicted": [...], "avg": 10.5},
    "amount": {"dates": [...], "predicted": [...], "avg": 15000}
  },
  "charts": {
    "quantity_forecast": "base64_image...",
    "amount_forecast": "base64_image..."
  },
  "insights": "LLM이 생성한 인사이트..."
}
```

### POST `/api/chat`
챗봇 대화

**Request:**
```json
{
  "file_id": "file_001",
  "user_id": "user123",
  "question": "이번 주 수량 예측이 어떻게 되나요?"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "이번 주 수량 예측은 평균 10.5개로..."
}
```

---

## 💡 주요 기능

### ✅ 자동 예측
- 수량 예측 모델 자동 학습
- 금액 예측 모델 자동 학습
- 향후 7일 예측

### ✅ 시각화
- 수량 예측 차트
- 금액 예측 차트
- 피처 중요도 차트
- 성능 대시보드

### ✅ LLM 챗봇
- 자동 인사이트 생성
- 사용자 질문 답변
- 대화 이력 저장

---

## 🔧 설정

### LLM 제공자 선택

```python
# OpenAI 사용
llm_service = LLMChatbotService(
    api_key="sk-...",
    provider="openai"
)

# Anthropic Claude 사용
llm_service = LLMChatbotService(
    api_key="sk-ant-...",
    provider="anthropic"
)
```

### MongoDB 연결

```python
from mongodb_schema import MongoDBService

mongo = MongoDBService(
    connection_string="mongodb://localhost:27017/",
    db_name="blinkit_analytics"
)
```

---

## 📊 예시 출력

### 예측 결과
```
수량 예측: 평균 10.5개/일
금액 예측: 평균 15,000원/일
정확도: 72.5%
```

### LLM 인사이트
```
안녕하세요! 수요 예측 분석 결과를 요약해드리겠습니다.

📊 주요 인사이트

1. 수량 예측: 향후 7일 평균 10.5개로 예상됩니다.
   - 예측 정확도: 72.5%
   - 이는 전반적으로 안정적인 판매 패턴을 보여줍니다.

2. 금액 예측: 평균 15,000원으로 예상됩니다.
   - 예측 정확도: 75.0%
   - 수량 대비 금액 증가율을 모니터링하세요.

3. 권장사항:
   - 재고는 평균 수량의 1.2배 정도 준비하시는 것을 권장합니다.
   - 주말 판매량이 증가하는 패턴이 보이므로 주말 재고를 늘려보세요.
```

---

## 🐛 문제 해결

### MongoDB 연결 오류
```bash
# MongoDB가 실행 중인지 확인
mongosh
```

### LLM API 오류
```bash
# API 키 확인
echo $OPENAI_API_KEY
```

### 시각화 생성 오류
```bash
# kaleido 설치 확인
pip install kaleido
```

---

## 📚 참고 문서

- [챗봇_시스템_설계.md](챗봇_시스템_설계.md) - 상세 설계 문서
- [MongoDB_설계_문서.md](MongoDB_설계_문서.md) - 데이터베이스 설계
- [README_자동화.md](README_자동화.md) - 자동화 파이프라인

---

*문서 작성일: 2024-12-29*

