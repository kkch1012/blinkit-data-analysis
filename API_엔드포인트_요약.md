# 📡 API 엔드포인트 요약

## 🔐 인증

| Method | Endpoint | 설명 |
|---------|----------|------|
| POST | `/api/auth/register` | 회원가입 |
| POST | `/api/auth/login` | 로그인 |

---

## 📁 파일 관리

| Method | Endpoint | 설명 |
|---------|----------|------|
| POST | `/api/files/upload` | CSV 파일 업로드 |
| GET | `/api/files` | 파일 목록 조회 |
| GET | `/api/files/<file_id>` | 파일 상세 정보 |

---

## 📊 시각화 및 통계

| Method | Endpoint | 설명 |
|---------|----------|------|
| POST | `/api/visualizations/statistics` | 통계 및 시각화 생성 |
| GET | `/api/visualizations/<file_id>` | 저장된 시각화 조회 |

### POST `/api/visualizations/statistics`
**Request:**
```json
{
  "file_id": "file_001",
  "options": {
    "include_charts": true,
    "chart_types": ["bar", "line", "pie"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_rows": 5000,
    "numeric_summary": {...},
    "categorical_summary": {...}
  },
  "charts": {
    "distribution": "base64_image...",
    "trend": "base64_image..."
  }
}
```

---

## 🔗 상관관계 분석

| Method | Endpoint | 설명 |
|---------|----------|------|
| POST | `/api/correlations/analyze` | 상관관계 분석 및 시각화 |
| GET | `/api/correlations/<file_id>` | 저장된 상관관계 결과 조회 |

### POST `/api/correlations/analyze`
**Request:**
```json
{
  "file_id": "file_001",
  "target_column": "수량",
  "features": ["금액", "평점", "temp_max", "rainfall"]
}
```

**Response:**
```json
{
  "success": true,
  "correlation_matrix": {
    "수량": {
      "금액": 0.85,
      "평점": 0.32,
      "temp_max": 0.15,
      "rainfall": 0.08
    }
  },
  "top_correlations": [
    {"feature": "금액", "correlation": 0.85}
  ],
  "chart": "base64_image...",
  "weights": {
    "금액": 0.45,
    "평점": 0.20,
    "temp_max": 0.15,
    "rainfall": 0.10
  }
}
```

---

## 🔮 예측

| Method | Endpoint | 설명 |
|---------|----------|------|
| POST | `/api/predictions/forecast` | 수량/금액 예측 |
| GET | `/api/predictions/<file_id>` | 저장된 예측 결과 조회 |

### POST `/api/predictions/forecast`
**Request:**
```json
{
  "file_id": "file_001",
  "target_columns": ["수량", "금액"],
  "forecast_days": 7,
  "weights": {
    "금액": 0.45,
    "평점": 0.20,
    "temp_max": 0.15
  }
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "quantity": {
      "dates": ["2024-12-30", "2024-12-31", ...],
      "predicted": [10.5, 12.3, ...],
      "avg": 10.5,
      "confidence_interval": {"lower": 8.5, "upper": 12.5}
    },
    "amount": {
      "dates": ["2024-12-30", "2024-12-31", ...],
      "predicted": [15000, 18000, ...],
      "avg": 15000,
      "confidence_interval": {"lower": 12000, "upper": 18000}
    }
  },
  "metrics": {
    "quantity": {"mae": 1.23, "r2": 0.65, "accuracy": 72.5},
    "amount": {"mae": 500.5, "r2": 0.70, "accuracy": 75.0}
  },
  "chart": "base64_image..."
}
```

---

## 🤖 LLM 솔루션

| Method | Endpoint | 설명 |
|---------|----------|------|
| POST | `/api/solutions/generate` | 모든 분석 결과를 LLM에 전달하여 솔루션 생성 |
| GET | `/api/solutions/<file_id>` | 저장된 솔루션 조회 |

### POST `/api/solutions/generate`
**Request:**
```json
{
  "file_id": "file_001",
  "include_visualizations": true,
  "include_correlations": true,
  "include_predictions": true
}
```

**Response:**
```json
{
  "success": true,
  "solution": {
    "summary": "분석 결과 요약...",
    "insights": [
      "수량과 금액 간 강한 양의 상관관계(0.85) 발견",
      "기온이 높을수록 판매량 증가 경향",
      "주말 판매량이 평일 대비 15% 높음"
    ],
    "recommendations": [
      "재고는 평균 수량의 1.2배 준비 권장",
      "주말 재고 증가 필요",
      "기온이 높은 날 마케팅 강화"
    ],
    "action_items": [
      {
        "priority": "high",
        "action": "주말 재고 20% 증가",
        "reason": "주말 판매량 증가 패턴 확인"
      },
      {
        "priority": "medium",
        "action": "기온 기반 동적 가격 조정",
        "reason": "기온과 판매량 상관관계 확인"
      }
    ]
  }
}
```

---

## 🔄 전체 플로우

```
1. POST /api/auth/register (회원가입)
   → user_id, token 받음

2. POST /api/files/upload (CSV 업로드)
   → file_id 받음

3. POST /api/visualizations/statistics (통계/시각화)
   → 통계 + 차트 받음

4. POST /api/correlations/analyze (상관관계 분석)
   → 상관계수 + 가중치 받음

5. POST /api/predictions/forecast (예측)
   → 수량/금액 예측 받음
   (weights는 4번에서 받은 값 사용)

6. POST /api/solutions/generate (LLM 솔루션)
   → 종합 솔루션 받음
   (1-5번의 모든 결과를 LLM에 전달)
```

---

## 📝 사용 예시

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:5000/api"
headers = {"Authorization": "Bearer your_token"}

# 1. CSV 업로드
with open('data.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{BASE_URL}/files/upload", files=files, headers=headers)
    file_id = response.json()['file_id']

# 2. 통계 및 시각화
response = requests.post(
    f"{BASE_URL}/visualizations/statistics",
    json={"file_id": file_id},
    headers=headers
)
stats = response.json()

# 3. 상관관계 분석
response = requests.post(
    f"{BASE_URL}/correlations/analyze",
    json={
        "file_id": file_id,
        "target_column": "수량",
        "features": ["금액", "평점", "temp_max"]
    },
    headers=headers
)
correlations = response.json()
weights = correlations['weights']

# 4. 예측
response = requests.post(
    f"{BASE_URL}/predictions/forecast",
    json={
        "file_id": file_id,
        "target_columns": ["수량", "금액"],
        "forecast_days": 7,
        "weights": weights
    },
    headers=headers
)
predictions = response.json()

# 5. LLM 솔루션
response = requests.post(
    f"{BASE_URL}/solutions/generate",
    json={
        "file_id": file_id,
        "include_visualizations": True,
        "include_correlations": True,
        "include_predictions": True
    },
    headers=headers
)
solution = response.json()
print(solution['solution'])
```

### JavaScript (fetch)

```javascript
const BASE_URL = 'http://localhost:5000/api';
const headers = {
  'Authorization': 'Bearer your_token',
  'Content-Type': 'application/json'
};

// 1. CSV 업로드
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch(`${BASE_URL}/files/upload`, {
  method: 'POST',
  headers: {'Authorization': 'Bearer your_token'},
  body: formData
});
const {file_id} = await uploadResponse.json();

// 2. 통계 및 시각화
const statsResponse = await fetch(`${BASE_URL}/visualizations/statistics`, {
  method: 'POST',
  headers,
  body: JSON.stringify({file_id})
});
const stats = await statsResponse.json();

// 3. 상관관계 분석
const corrResponse = await fetch(`${BASE_URL}/correlations/analyze`, {
  method: 'POST',
  headers,
  body: JSON.stringify({
    file_id,
    target_column: '수량',
    features: ['금액', '평점', 'temp_max']
  })
});
const correlations = await corrResponse.json();
const weights = correlations.weights;

// 4. 예측
const predResponse = await fetch(`${BASE_URL}/predictions/forecast`, {
  method: 'POST',
  headers,
  body: JSON.stringify({
    file_id,
    target_columns: ['수량', '금액'],
    forecast_days: 7,
    weights
  })
});
const predictions = await predResponse.json();

// 5. LLM 솔루션
const solutionResponse = await fetch(`${BASE_URL}/solutions/generate`, {
  method: 'POST',
  headers,
  body: JSON.stringify({
    file_id,
    include_visualizations: true,
    include_correlations: true,
    include_predictions: true
  })
});
const solution = await solutionResponse.json();
console.log(solution.solution);
```

---

## ⚠️ 에러 응답 형식

모든 에러는 다음 형식으로 반환:

```json
{
  "success": false,
  "error": "에러 메시지"
}
```

**HTTP 상태 코드:**
- `200`: 성공
- `400`: 잘못된 요청 (필수 파라미터 누락 등)
- `401`: 인증 실패
- `404`: 리소스를 찾을 수 없음
- `500`: 서버 오류

---

*문서 작성일: 2024-12-29*

