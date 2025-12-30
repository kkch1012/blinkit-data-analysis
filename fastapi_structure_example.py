"""
FastAPI + MongoDB 도메인별 구조 예시 코드
실제 구현 시 이 구조를 참고하여 작성
"""

# ============================================
# app/main.py
# ============================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import correlations, predictions, solutions
from app.core.config import settings
from app.core.database import init_db, close_db

app = FastAPI(
    title="Blinkit Analytics API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(correlations.router, prefix="/api/v1/correlations", tags=["correlations"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(solutions.router, prefix="/api/v1/solutions", tags=["solutions"])

@app.on_event("startup")
async def startup_event():
    await init_db()

@app.on_event("shutdown")
async def shutdown_event():
    await close_db()


# ============================================
# app/core/database.py
# ============================================
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

class Database:
    client: AsyncIOMotorClient = None

database = Database()

async def get_database():
    """데이터베이스 인스턴스 반환"""
    return database.client[settings.DATABASE_NAME]

async def init_db():
    """MongoDB 연결 초기화"""
    database.client = AsyncIOMotorClient(settings.MONGODB_URL)
    await database.client.admin.command('ping')
    print("✅ MongoDB 연결 성공")

async def close_db():
    """MongoDB 연결 종료"""
    if database.client:
        database.client.close()


# ============================================
# app/services/correlation/correlation_service.py
# ============================================
from typing import Dict, List
from app.core.database import get_database
from app.models.correlation import CorrelationAnalysisRequest, CorrelationAnalysisResponse
from app.services.correlation.weight_calculator import WeightCalculator
from app.services.correlation.correlation_repository import CorrelationRepository
import pandas as pd
import numpy as np

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
        """상관관계 분석 및 가중치 계산"""
        # 1. 데이터 로드
        data = await self._load_data(file_id)
        df = pd.DataFrame(data)
        
        # 2. 상관계수 계산
        correlations = {}
        for feature in features:
            if feature in df.columns:
                corr = df[[target_column, feature]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations[feature] = float(corr)
        
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
    
    async def _load_data(self, file_id: str) -> List[Dict]:
        """MongoDB에서 데이터 로드"""
        db = await get_database()
        collection = db['csv_contents']
        cursor = collection.find({"file_id": file_id}).sort("row_index", 1)
        docs = await cursor.to_list(length=None)
        return [doc['data'] for doc in docs]
    
    async def _create_chart(self, correlations: Dict, target: str) -> str:
        """차트 생성 (Base64)"""
        import plotly.graph_objects as go
        import base64
        from io import BytesIO
        
        features = list(correlations.keys())
        values = list(correlations.values())
        
        fig = go.Figure(data=[
            go.Bar(x=features, y=values, marker_color='steelblue')
        ])
        fig.update_layout(
            title=f'{target}와의 상관관계',
            xaxis_title='피처',
            yaxis_title='상관계수'
        )
        
        img_buffer = BytesIO()
        fig.write_image(img_buffer, format='png')
        return base64.b64encode(img_buffer.getvalue()).decode()
    
    def _get_top_correlations(self, correlations: Dict, top_n: int = 5) -> List[Dict]:
        """상위 상관관계 추출"""
        sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        return [
            {"feature": k, "correlation": v}
            for k, v in sorted_items[:top_n]
        ]
    
    async def get_correlations(self, file_id: str) -> CorrelationAnalysisResponse:
        """저장된 상관관계 조회"""
        result = await self.repository.get_by_file_id(file_id)
        if not result:
            return None
        
        return CorrelationAnalysisResponse(
            correlation_matrix={result['target_column']: result['correlation_matrix']},
            top_correlations=self._get_top_correlations(result['correlation_matrix']),
            chart=result['chart'],
            weights=result['weights'],
            correlation_id=result['correlation_id']
        )


# ============================================
# app/services/correlation/weight_calculator.py
# ============================================
from typing import Dict

class WeightCalculator:
    """가중치 계산기"""
    
    def calculate(self, correlations: Dict[str, float]) -> Dict[str, float]:
        """상관계수 기반 가중치 계산"""
        abs_correlations = {k: abs(v) for k, v in correlations.items()}
        total = sum(abs_correlations.values())
        
        if total > 0:
            weights = {k: v/total for k, v in abs_correlations.items()}
        else:
            weights = {k: 1/len(abs_correlations) for k in abs_correlations.keys()}
        
        return weights


# ============================================
# app/services/correlation/correlation_repository.py
# ============================================
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


# ============================================
# app/api/v1/correlations.py
# ============================================
from fastapi import APIRouter, Depends, HTTPException
from app.models.correlation import CorrelationAnalysisRequest, CorrelationAnalysisResponse
from app.services.correlation.correlation_service import CorrelationService
from app.api.deps import get_current_user

router = APIRouter()

def get_correlation_service() -> CorrelationService:
    """의존성 주입"""
    return CorrelationService()

@router.post("/analyze", response_model=CorrelationAnalysisResponse)
async def analyze_correlations(
    request: CorrelationAnalysisRequest,
    current_user: dict = Depends(get_current_user),
    service: CorrelationService = Depends(get_correlation_service)
):
    """상관관계 분석 및 시각화"""
    try:
        result = await service.analyze_correlations(
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
    service: CorrelationService = Depends(get_correlation_service)
):
    """저장된 상관관계 분석 결과 조회"""
    result = await service.get_correlations(file_id)
    if not result:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다")
    return result


# ============================================
# app/services/solution/solution_service.py
# ============================================
from typing import Dict
from app.core.database import get_database
from app.services.visualization.visualization_service import VisualizationService
from app.services.correlation.correlation_service import CorrelationService
from app.services.prediction.prediction_service import PredictionService
from app.services.solution.llm_service import LLMService

class SolutionService:
    """솔루션 생성 서비스"""
    
    def __init__(self):
        self.viz_service = VisualizationService()
        self.corr_service = CorrelationService()
        self.pred_service = PredictionService()
        self.llm_service = LLMService()
    
    async def generate_solution(
        self,
        file_id: str,
        user_id: str,
        include_visualizations: bool = True,
        include_correlations: bool = True,
        include_predictions: bool = True
    ) -> Dict:
        """모든 분석 결과를 수집하여 LLM 솔루션 생성"""
        context = {}
        
        # 1. 시각화 데이터 수집
        if include_visualizations:
            viz = await self.viz_service.get_visualizations(file_id)
            if viz:
                context['visualizations'] = {
                    'statistics': viz.get('statistics', {}),
                    'summary': '시각화 데이터가 있습니다'
                }
        
        # 2. 상관관계 데이터 수집
        if include_correlations:
            corr = await self.corr_service.get_correlations(file_id)
            if corr:
                context['correlations'] = {
                    'top_correlations': corr.top_correlations,
                    'weights': corr.weights
                }
        
        # 3. 예측 데이터 수집
        if include_predictions:
            pred = await self.pred_service.get_predictions(file_id)
            if pred:
                context['predictions'] = {
                    'quantity': pred.predictions.get('quantity', {}),
                    'amount': pred.predictions.get('amount', {}),
                    'metrics': pred.metrics
                }
        
        # 4. LLM 솔루션 생성
        solution = await self.llm_service.generate_solution(context)
        
        # 5. 저장
        await self._save_solution(file_id, user_id, solution)
        
        return solution
    
    async def _save_solution(self, file_id: str, user_id: str, solution: Dict):
        """솔루션 저장"""
        db = await get_database()
        collection = db['solutions']
        
        doc = {
            'solution_id': f"sol_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'file_id': file_id,
            'user_id': user_id,
            'solution': solution,
            'created_at': datetime.now()
        }
        
        await collection.insert_one(doc)


# ============================================
# app/services/solution/llm_service.py
# ============================================
from typing import Dict
import os
import json

class LLMService:
    """LLM 서비스"""
    
    def __init__(self):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except:
            self.client = None
    
    async def generate_solution(self, context: Dict) -> Dict:
        """LLM 솔루션 생성"""
        prompt = f"""
당신은 데이터 분석 전문가입니다. 다음 분석 결과를 바탕으로 종합 솔루션을 제공해주세요.

[분석 결과]
{json.dumps(context, ensure_ascii=False, indent=2)}

위 결과를 바탕으로 다음 형식으로 답변해주세요:
1. **요약**: 전체 분석 결과를 한 문단으로 요약
2. **주요 인사이트**: 3-5개의 핵심 발견사항
3. **권장사항**: 구체적인 행동 권장사항
4. **실행 항목**: 우선순위별 실행 항목 리스트

한국어로 전문적이고 친절하게 작성해주세요.
"""
        
        if self.client:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            solution_text = response.choices[0].message.content
        else:
            solution_text = self._mock_solution(context)
        
        return self._parse_solution(solution_text)
    
    def _parse_solution(self, text: str) -> Dict:
        """LLM 응답을 구조화"""
        return {
            'summary': text[:200] if len(text) > 200 else text,
            'insights': self._extract_list(text, '인사이트'),
            'recommendations': self._extract_list(text, '권장사항'),
            'action_items': self._extract_action_items(text)
        }
    
    def _extract_list(self, text: str, keyword: str) -> list:
        """리스트 추출"""
        items = []
        if keyword in text:
            section = text.split(keyword)[1].split('**')[0] if '**' in text else text
            for line in section.split('\n'):
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    items.append(line.strip().lstrip('-•').strip())
        return items[:5]
    
    def _extract_action_items(self, text: str) -> list:
        """실행 항목 추출"""
        items = []
        if '실행 항목' in text:
            section = text.split('실행 항목')[1]
            for line in section.split('\n'):
                if 'high' in line.lower() or '중요' in line:
                    items.append({'priority': 'high', 'action': line.strip()})
                elif 'medium' in line.lower() or '보통' in line:
                    items.append({'priority': 'medium', 'action': line.strip()})
        return items
    
    def _mock_solution(self, context: Dict) -> str:
        """모의 솔루션"""
        return """
**요약**: 분석 결과를 종합하면, 데이터에서 중요한 패턴이 발견되었습니다.

**주요 인사이트**:
- 상관관계 분석 결과 주요 변수들이 확인되었습니다
- 예측 모델의 정확도가 양호한 수준입니다

**권장사항**:
- 예측 결과를 바탕으로 재고 관리 최적화
- 주요 변수 모니터링 강화

**실행 항목**:
- [high] 재고 수준 조정
- [medium] 마케팅 전략 수정
"""


# ============================================
# app/models/correlation.py
# ============================================
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class CorrelationAnalysisRequest(BaseModel):
    """상관관계 분석 요청"""
    file_id: str = Field(..., description="파일 ID")
    target_column: str = Field(..., description="타겟 컬럼명")
    features: List[str] = Field(..., description="분석할 피처 리스트")

class CorrelationAnalysisResponse(BaseModel):
    """상관관계 분석 응답"""
    correlation_matrix: Dict[str, Dict[str, float]]
    top_correlations: List[Dict[str, float]]
    chart: str
    weights: Dict[str, float]
    correlation_id: Optional[str] = None


# ============================================
# app/api/deps.py
# ============================================
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from app.core.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """현재 사용자 조회"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증 정보를 확인할 수 없습니다",
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return {"user_id": user_id}
    except JWTError:
        raise credentials_exception

