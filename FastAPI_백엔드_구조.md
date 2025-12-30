# 🚀 FastAPI + MongoDB 최적 백엔드 구조

## 📋 설계 원칙

1. **도메인 주도 설계 (DDD)**: 각 도메인별로 독립적인 구조
2. **의존성 주입**: FastAPI의 DI 시스템 활용
3. **비동기 지원**: MongoDB 비동기 드라이버 사용
4. **계층 분리**: API → Service → Repository → Model
5. **확장성**: 새로운 도메인 추가가 쉬운 구조

---

## 🗂️ 최적 디렉토리 구조

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI 앱 초기화
│   ├── config.py                    # 설정 관리
│   ├── dependencies.py              # 공통 의존성
│   │
│   ├── api/                         # API 라우터 (프레젠테이션 레이어)
│   │   ├── __init__.py
│   │   ├── v1/                      # API 버전 관리
│   │   │   ├── __init__.py
│   │   │   ├── auth.py              # 인증 라우터
│   │   │   ├── users.py             # 유저 라우터
│   │   │   ├── files.py             # 파일 라우터
│   │   │   ├── visualizations.py    # 시각화 라우터
│   │   │   ├── correlations.py     # 상관관계 라우터
│   │   │   ├── predictions.py       # 예측 라우터
│   │   │   └── solutions.py         # 솔루션 라우터
│   │   └── deps.py                  # API 의존성
│   │
│   ├── core/                        # 핵심 설정
│   │   ├── __init__.py
│   │   ├── security.py              # JWT, 비밀번호 해싱
│   │   ├── database.py              # MongoDB 연결
│   │   └── exceptions.py            # 커스텀 예외
│   │
│   ├── models/                       # Pydantic 모델 (스키마)
│   │   ├── __init__.py
│   │   ├── common.py                # 공통 모델
│   │   ├── user.py                  # 유저 모델
│   │   ├── file.py                  # 파일 모델
│   │   ├── visualization.py        # 시각화 모델
│   │   ├── correlation.py           # 상관관계 모델
│   │   ├── prediction.py            # 예측 모델
│   │   └── solution.py              # 솔루션 모델
│   │
│   ├── schemas/                     # MongoDB 문서 스키마 (선택적)
│   │   ├── __init__.py
│   │   ├── user_schema.py
│   │   └── file_schema.py
│   │
│   ├── services/                    # 비즈니스 로직 (도메인별 분리)
│   │   ├── __init__.py
│   │   │
│   │   ├── auth/                    # 인증 도메인
│   │   │   ├── __init__.py
│   │   │   ├── auth_service.py
│   │   │   └── token_service.py
│   │   │
│   │   ├── user/                    # 유저 도메인
│   │   │   ├── __init__.py
│   │   │   ├── user_service.py
│   │   │   └── user_repository.py
│   │   │
│   │   ├── file/                    # 파일 도메인
│   │   │   ├── __init__.py
│   │   │   ├── file_service.py
│   │   │   └── file_repository.py
│   │   │
│   │   ├── visualization/           # 시각화 도메인
│   │   │   ├── __init__.py
│   │   │   ├── visualization_service.py
│   │   │   ├── chart_generator.py
│   │   │   └── statistics_calculator.py
│   │   │
│   │   ├── correlation/             # 상관관계 도메인
│   │   │   ├── __init__.py
│   │   │   ├── correlation_service.py
│   │   │   ├── weight_calculator.py
│   │   │   └── correlation_repository.py
│   │   │
│   │   ├── prediction/              # 예측 도메인
│   │   │   ├── __init__.py
│   │   │   ├── prediction_service.py
│   │   │   ├── model_trainer.py
│   │   │   └── forecast_generator.py
│   │   │
│   │   └── solution/                # 솔루션 도메인
│   │       ├── __init__.py
│   │       ├── solution_service.py
│   │       └── llm_service.py
│   │
│   ├── repositories/                # 데이터 접근 레이어 (선택적)
│   │   ├── __init__.py
│   │   ├── base_repository.py       # 기본 CRUD
│   │   ├── user_repository.py
│   │   ├── file_repository.py
│   │   └── ...
│   │
│   └── utils/                       # 유틸리티
│       ├── __init__.py
│       ├── validators.py
│       ├── helpers.py
│       └── constants.py
│
├── tests/                           # 테스트
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api/
│   ├── test_services/
│   └── test_utils/
│
├── requirements.txt
├── .env
├── .env.example
└── README.md
```

---

## 📝 핵심 파일 구조

### 1. `app/main.py` - FastAPI 앱 초기화

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import auth, users, files, visualizations, correlations, predictions, solutions
from app.core.config import settings
from app.core.database import init_db

app = FastAPI(
    title="Blinkit Analytics API",
    version="1.0.0",
    description="수요 예측 및 분석 API"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(files.router, prefix="/api/v1/files", tags=["files"])
app.include_router(visualizations.router, prefix="/api/v1/visualizations", tags=["visualizations"])
app.include_router(correlations.router, prefix="/api/v1/correlations", tags=["correlations"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(solutions.router, prefix="/api/v1/solutions", tags=["solutions"])

@app.on_event("startup")
async def startup_event():
    await init_db()

@app.get("/")
async def root():
    return {"message": "Blinkit Analytics API"}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### 2. `app/core/database.py` - MongoDB 연결

```python
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

class Database:
    client: AsyncIOMotorClient = None

database = Database()

async def get_database():
    return database.client[settings.DATABASE_NAME]

async def init_db():
    """MongoDB 연결 초기화"""
    database.client = AsyncIOMotorClient(settings.MONGODB_URL)
    # 연결 테스트
    await database.client.admin.command('ping')
    print("✅ MongoDB 연결 성공")

async def close_db():
    """MongoDB 연결 종료"""
    if database.client:
        database.client.close()
```

### 3. `app/core/config.py` - 설정 관리

```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "blinkit_analytics"
    
    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # LLM
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 4. `app/core/security.py` - 인증

```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt
```

---

## 🏗️ 도메인별 서비스 구조 예시

### `app/services/correlation/correlation_service.py`

```python
from typing import Dict, List
from app.core.database import get_database
from app.models.correlation import CorrelationAnalysisRequest, CorrelationAnalysisResponse
from app.services.correlation.weight_calculator import WeightCalculator
from app.services.correlation.correlation_repository import CorrelationRepository

class CorrelationService:
    """상관관계 분석 서비스"""
    
    def __init__(self):
        self.weight_calculator = WeightCalculator()
        self.repository = CorrelationRepository()
    
    async def analyze_correlations(
        self, 
        file_id: str,
        target_column: str,
        features: List[str],
        user_id: str
    ) -> CorrelationAnalysisResponse:
        """
        상관관계 분석 및 가중치 계산
        """
        # 1. 데이터 로드
        db = await get_database()
        data = await self._load_data(db, file_id)
        
        # 2. 상관계수 계산
        correlations = await self._calculate_correlations(
            data, target_column, features
        )
        
        # 3. 가중치 계산
        weights = self.weight_calculator.calculate(correlations)
        
        # 4. 시각화 생성
        chart = await self._create_chart(correlations, target_column)
        
        # 5. 결과 저장
        result = await self.repository.save(
            file_id=file_id,
            user_id=user_id,
            target_column=target_column,
            correlations=correlations,
            weights=weights,
            chart=chart
        )
        
        return CorrelationAnalysisResponse(
            correlation_matrix={target_column: correlations},
            top_correlations=self._get_top_correlations(correlations),
            chart=chart,
            weights=weights,
            correlation_id=result['correlation_id']
        )
    
    async def _load_data(self, db, file_id: str):
        """MongoDB에서 데이터 로드"""
        collection = db['csv_contents']
        cursor = collection.find({"file_id": file_id}).sort("row_index", 1)
        data = await cursor.to_list(length=None)
        return [doc['data'] for doc in data]
    
    async def _calculate_correlations(self, data, target: str, features: List[str]) -> Dict[str, float]:
        """상관계수 계산"""
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame(data)
        correlations = {}
        
        for feature in features:
            if feature in df.columns:
                corr = df[[target, feature]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations[feature] = float(corr)
        
        return correlations
    
    async def _create_chart(self, correlations: Dict, target: str) -> str:
        """차트 생성"""
        # Plotly로 차트 생성 후 Base64 변환
        pass
    
    def _get_top_correlations(self, correlations: Dict, top_n: int = 5) -> List[Dict]:
        """상위 상관관계 추출"""
        sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        return [
            {"feature": k, "correlation": v}
            for k, v in sorted_items[:top_n]
        ]
```

### `app/services/correlation/weight_calculator.py`

```python
from typing import Dict

class WeightCalculator:
    """가중치 계산기"""
    
    def calculate(self, correlations: Dict[str, float]) -> Dict[str, float]:
        """
        상관계수 기반 가중치 계산
        """
        # 절댓값 사용
        abs_correlations = {k: abs(v) for k, v in correlations.items()}
        
        # 정규화 (합이 1이 되도록)
        total = sum(abs_correlations.values())
        if total > 0:
            weights = {k: v/total for k, v in abs_correlations.items()}
        else:
            weights = {k: 1/len(abs_correlations) for k in abs_correlations.keys()}
        
        return weights
```

### `app/services/correlation/correlation_repository.py`

```python
from typing import Dict, Optional
from datetime import datetime
from app.core.database import get_database

class CorrelationRepository:
    """상관관계 데이터 접근 레이어"""
    
    async def save(
        self,
        file_id: str,
        user_id: str,
        target_column: str,
        correlations: Dict[str, float],
        weights: Dict[str, float],
        chart: str
    ) -> Dict:
        """상관관계 분석 결과 저장"""
        db = await get_database()
        collection = db['correlations']
        
        doc = {
            'correlation_id': f"corr_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'file_id': file_id,
            'user_id': user_id,
            'target_column': target_column,
            'correlation_matrix': correlations,
            'weights': weights,
            'chart': chart,
            'created_at': datetime.now()
        }
        
        result = await collection.insert_one(doc)
        doc['_id'] = result.inserted_id
        return doc
    
    async def get_by_file_id(self, file_id: str) -> Optional[Dict]:
        """파일 ID로 조회"""
        db = await get_database()
        collection = db['correlations']
        return await collection.find_one(
            {'file_id': file_id},
            sort=[('created_at', -1)]
        )
```

---

## 🔌 API 라우터 예시

### `app/api/v1/correlations.py`

```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.models.correlation import CorrelationAnalysisRequest, CorrelationAnalysisResponse
from app.services.correlation.correlation_service import CorrelationService
from app.api.deps import get_current_user

router = APIRouter()

@router.post("/analyze", response_model=CorrelationAnalysisResponse)
async def analyze_correlations(
    request: CorrelationAnalysisRequest,
    current_user: dict = Depends(get_current_user),
    correlation_service: CorrelationService = Depends()
):
    """
    상관관계 분석 및 시각화
    """
    try:
        result = await correlation_service.analyze_correlations(
            file_id=request.file_id,
            target_column=request.target_column,
            features=request.features,
            user_id=current_user['user_id']
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{file_id}", response_model=CorrelationAnalysisResponse)
async def get_correlations(
    file_id: str,
    current_user: dict = Depends(get_current_user),
    correlation_service: CorrelationService = Depends()
):
    """저장된 상관관계 분석 결과 조회"""
    result = await correlation_service.get_correlations(file_id)
    if not result:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다")
    return result
```

### `app/api/deps.py` - 공통 의존성

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from app.core.config import settings
from app.services.user.user_service import UserService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """현재 사용자 조회 (JWT 검증)"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증 정보를 확인할 수 없습니다",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user_service = UserService()
    user = await user_service.get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    return user
```

---

## 📦 Pydantic 모델 예시

### `app/models/correlation.py`

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class CorrelationAnalysisRequest(BaseModel):
    """상관관계 분석 요청"""
    file_id: str = Field(..., description="파일 ID")
    target_column: str = Field(..., description="타겟 컬럼명")
    features: List[str] = Field(..., description="분석할 피처 리스트")

class CorrelationAnalysisResponse(BaseModel):
    """상관관계 분석 응답"""
    correlation_matrix: Dict[str, Dict[str, float]] = Field(..., description="상관관계 행렬")
    top_correlations: List[Dict[str, float]] = Field(..., description="상위 상관관계")
    chart: str = Field(..., description="차트 이미지 (Base64)")
    weights: Dict[str, float] = Field(..., description="피처 가중치")
    correlation_id: Optional[str] = Field(None, description="저장된 분석 ID")
```

---

## 🔄 의존성 주입 패턴

### 서비스 의존성 주입

```python
# app/api/v1/correlations.py
from app.services.correlation.correlation_service import CorrelationService

def get_correlation_service() -> CorrelationService:
    """상관관계 서비스 의존성"""
    return CorrelationService()

@router.post("/analyze")
async def analyze_correlations(
    request: CorrelationAnalysisRequest,
    service: CorrelationService = Depends(get_correlation_service)
):
    result = await service.analyze_correlations(...)
    return result
```

---

## ✅ 도메인별 구조의 장점

### 1. 관심사 분리
```
correlation/
  ├── correlation_service.py    # 비즈니스 로직
  ├── weight_calculator.py      # 가중치 계산 (단일 책임)
  └── correlation_repository.py # 데이터 접근
```

### 2. 확장성
- 새로운 도메인 추가 시 `services/new_domain/` 폴더만 추가
- 기존 코드에 영향 없음

### 3. 테스트 용이성
```python
# tests/test_services/correlation/test_correlation_service.py
from app.services.correlation.correlation_service import CorrelationService

async def test_analyze_correlations():
    service = CorrelationService()
    # 테스트 코드
```

### 4. 재사용성
- `weight_calculator.py`는 다른 도메인에서도 사용 가능
- 공통 로직을 별도 모듈로 분리

---

## 🚀 실행 구조

### `run.py`

```python
import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

---

## 📦 requirements.txt

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
motor==3.3.2              # MongoDB 비동기 드라이버
pymongo==4.6.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
plotly==5.18.0
kaleido==0.2.1
openai==1.3.0
python-dotenv==1.0.0
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
```

---

## 🔄 전체 플로우 예시

```
[API 요청]
POST /api/v1/correlations/analyze
  ↓
[라우터]
app/api/v1/correlations.py
  ↓
[서비스]
app/services/correlation/correlation_service.py
  ├─ weight_calculator.py (가중치 계산)
  └─ correlation_repository.py (데이터 저장)
  ↓
[응답]
CorrelationAnalysisResponse
```

---

## 💡 Best Practices

### 1. 비동기 사용
```python
# ✅ 좋은 예
async def analyze_correlations(...):
    db = await get_database()
    data = await collection.find_one(...)

# ❌ 나쁜 예
def analyze_correlations(...):
    db = get_database()  # 동기 방식
```

### 2. 의존성 주입
```python
# ✅ 좋은 예
@router.post("/analyze")
async def analyze(
    service: CorrelationService = Depends(get_correlation_service)
):
    pass

# ❌ 나쁜 예
@router.post("/analyze")
async def analyze():
    service = CorrelationService()  # 직접 생성
```

### 3. 에러 처리
```python
from app.core.exceptions import NotFoundError, ValidationError

async def get_correlations(file_id: str):
    result = await repository.get_by_file_id(file_id)
    if not result:
        raise NotFoundError(f"File {file_id} not found")
    return result
```

---

## 📊 구조 비교

| 구조 | 장점 | 단점 |
|------|------|------|
| **도메인별 분리** (추천) | 확장성, 유지보수성, 테스트 용이 | 초기 구조가 복잡할 수 있음 |
| 기능별 분리 | 단순함 | 도메인 간 의존성 증가 |
| 계층별 분리 | 명확한 계층 | 도메인 로직이 분산 |

---

*문서 작성일: 2024-12-29*

