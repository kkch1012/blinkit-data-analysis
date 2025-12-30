# 🚀 자동화 파이프라인 및 MongoDB 웹 서비스

## 📋 목차
1. [자동화 파이프라인](#자동화-파이프라인)
2. [MongoDB 스키마 설계](#mongodb-스키마-설계)
3. [웹 서비스 API](#웹-서비스-api)
4. [사용 예시](#사용-예시)

---

## 🔧 자동화 파이프라인

### 기능
- ✅ **자동 컬럼 감지**: 날짜, 범주형, 수치형 자동 분류
- ✅ **타겟 컬럼 자동 감지**: 수량, 금액 등 키워드 기반 감지
- ✅ **피처 엔지니어링 자동화**: 시계열, 날짜, 범주형 인코딩
- ✅ **가중치 자동 계산**: 상관계수 기반 가중치 생성
- ✅ **모델 자동 학습**: Random Forest 자동 학습 및 평가

### 사용법

```python
from auto_feature_pipeline import AutoFeaturePipeline

# 파이프라인 초기화
pipeline = AutoFeaturePipeline()

# CSV 처리 (어떤 CSV든 가능!)
result = pipeline.process_csv(
    csv_path='data/your_data.csv',
    group_by=['상품명', '지역']  # 선택적
)

# 결과 확인
print(f"정확도: {result['metrics']['accuracy']:.2f}%")
print(f"가중치: {result['weights']}")
```

### 자동 감지 기능

| 컬럼 타입 | 감지 키워드 | 생성 피처 |
|----------|------------|----------|
| 날짜 | date, 날짜, time, 시간 | month, day_of_week, is_weekend |
| 타겟 | 수량, quantity, 금액, amount | - |
| 범주형 | object 타입 또는 고유값 < 10% | Label Encoded |
| 수치형 | int64, float64 | 정규화 (선택적) |

---

## 🗄️ MongoDB 스키마 설계

### 컬렉션 구조

#### 1. `files` - 파일 메타데이터
```json
{
  "file_id": "file_20241229120000_user123",
  "user_id": "user123",
  "file_name": "blinkit_data.csv",
  "columns": ["주문날짜", "상품명", "수량"],
  "row_count": 5000,
  "uploaded_at": ISODate("2024-12-29T12:00:00Z")
}
```

#### 2. `csv_contents` - 실제 데이터
```json
{
  "file_id": "file_20241229120000_user123",
  "user_id": "user123",
  "row_index": 0,
  "data": {
    "주문날짜": "2024-07-17",
    "상품명": "Pet Treats",
    "수량": 3
  }
}
```

#### 3. `feature_weights` - 피처 가중치
```json
{
  "file_id": "file_20241229120000_user123",
  "weights": {
    "수량_lag_1": 0.25,
    "temp_max": 0.15
  },
  "model_metrics": {
    "mae": 1.23,
    "r2": 0.65,
    "accuracy": 72.5
  }
}
```

#### 4. `user_suggestions` - 자동 제안
```json
{
  "file_id": "file_20241229120000_user123",
  "suggestions": [
    "💰 '금액' 컬럼이 있네요! 합계/평균을 구해드릴까요?",
    "📦 '수량' 컬럼이 있네요! 총 판매량을 계산해드릴까요?"
  ]
}
```

### 인덱스
```javascript
// 파일 조회 최적화
db.files.createIndex({ "user_id": 1, "uploaded_at": -1 })
db.csv_contents.createIndex({ "file_id": 1, "row_index": 1 })
db.feature_weights.createIndex({ "file_id": 1 })
```

---

## 🌐 웹 서비스 API

### 엔드포인트

#### 1. CSV 업로드
```http
POST /api/upload
Content-Type: multipart/form-data

file: [CSV 파일]
user_id: user123
```

**Response:**
```json
{
  "file_id": "file_20241229120000_user123",
  "columns": ["주문날짜", "상품명", "수량"],
  "row_count": 5000,
  "suggestions": [
    "💰 '금액' 컬럼이 있네요! 합계/평균을 구해드릴까요?"
  ]
}
```

#### 2. 자동 분석
```http
POST /api/analyze
Content-Type: application/json

{
  "file_id": "file_20241229120000_user123",
  "target_column": "수량",
  "group_by": ["상품명", "지역"]
}
```

**Response:**
```json
{
  "analysis_id": "analysis_20241229120000",
  "metrics": {
    "mae": 1.23,
    "r2": 0.65,
    "accuracy": 72.5
  },
  "weights": {
    "수량_lag_1": 0.25,
    "temp_max": 0.15
  }
}
```

#### 3. 파일 목록 조회
```http
GET /api/files/{user_id}
```

#### 4. 피처 가중치 조회
```http
GET /api/weights/{file_id}
```

---

## 📝 사용 예시

### 예시 1: 로컬에서 자동화 파이프라인 사용

```python
from auto_feature_pipeline import AutoFeaturePipeline

# 파이프라인 생성
pipeline = AutoFeaturePipeline()

# 다른 CSV 파일 처리
result = pipeline.process_csv(
    csv_path='data/new_sales_data.csv',
    group_by=['product', 'region']
)

# 결과 확인
print(f"✅ 정확도: {result['metrics']['accuracy']:.2f}%")
print(f"📊 상위 5개 피처 가중치:")
for feat, weight in sorted(result['weights'].items(), 
                          key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {feat}: {weight:.4f}")
```

### 예시 2: MongoDB 서비스 사용

```python
from mongodb_schema import MongoDBService

# MongoDB 서비스 초기화
mongo = MongoDBService()

# CSV 업로드
result = mongo.upload_csv(
    user_id="user123",
    file_path="data/blinkit_data.csv",
    file_name="blinkit_data.csv",
    file_size=1024000
)

# 제안 확인
suggestions = mongo.get_suggestions(result['file_id'])
for suggestion in suggestions:
    print(f"💡 {suggestion}")
```

### 예시 3: 웹 서비스 통합

```python
# Flask 서버 실행
python web_service_integration.py

# 클라이언트에서 호출
import requests

# 1. 파일 업로드
files = {'file': open('data.csv', 'rb')}
data = {'user_id': 'user123'}
response = requests.post('http://localhost:5000/api/upload', 
                        files=files, data=data)
file_info = response.json()

# 2. 자동 분석
analysis_data = {
    'file_id': file_info['file_id'],
    'group_by': ['상품명', '지역']
}
response = requests.post('http://localhost:5000/api/analyze', 
                        json=analysis_data)
result = response.json()

print(f"정확도: {result['metrics']['accuracy']:.2f}%")
```

---

## 🔄 전체 웹 서비스 플로우

```
[1] 사용자 CSV 업로드
    ↓
[2] 서버가 CSV 파싱 → MongoDB에 저장
    ├─ files 컬렉션: 메타데이터
    └─ csv_contents 컬렉션: 실제 데이터
    ↓
[3] 자동 컬럼 분석 → 제안 생성
    └─ user_suggestions 컬렉션: "금액 합계 구할까요?"
    ↓
[4] 사용자가 "분석 시작" 클릭
    ↓
[5] 자동화 파이프라인 실행
    ├─ 피처 엔지니어링
    ├─ 가중치 계산
    └─ 모델 학습
    ↓
[6] 결과 저장
    ├─ analysis_results: 분석 결과
    └─ feature_weights: 피처 가중치
    ↓
[7] 사용자에게 결과 반환
```

---

## 📦 설치 및 실행

### 1. 의존성 설치
```bash
pip install pandas numpy scikit-learn pymongo flask
```

### 2. MongoDB 실행
```bash
# MongoDB 시작
mongod

# 또는 Docker 사용
docker run -d -p 27017:27017 mongo
```

### 3. 웹 서비스 실행
```bash
python web_service_integration.py
```

### 4. 테스트
```bash
# 파일 업로드 테스트
curl -X POST http://localhost:5000/api/upload \
  -F "file=@data/blinkit_data.csv" \
  -F "user_id=test_user"
```

---

## 🎯 주요 특징

✅ **완전 자동화**: 어떤 CSV든 자동으로 처리  
✅ **스마트 감지**: 컬럼 타입, 타겟 자동 감지  
✅ **가중치 자동 생성**: 상관계수 기반 가중치  
✅ **MongoDB 통합**: 웹 서비스용 완전한 스키마  
✅ **실시간 제안**: 파일 업로드 시 자동 제안 생성  

---

*문서 작성일: 2024-12-29*

