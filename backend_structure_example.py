"""
백엔드 구조 예시 코드
실제 구현 시 이 구조를 참고하여 작성
"""

# ============================================
# app/__init__.py
# ============================================
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Blueprint 등록
    from app.api import auth, files, visualizations, correlations, predictions, solutions
    
    app.register_blueprint(auth.bp, url_prefix='/api/auth')
    app.register_blueprint(files.bp, url_prefix='/api/files')
    app.register_blueprint(visualizations.bp, url_prefix='/api/visualizations')
    app.register_blueprint(correlations.bp, url_prefix='/api/correlations')
    app.register_blueprint(predictions.bp, url_prefix='/api/predictions')
    app.register_blueprint(solutions.bp, url_prefix='/api/solutions')
    
    return app


# ============================================
# app/api/correlations.py
# ============================================
from flask import Blueprint, request, jsonify
from services.correlation_service import CorrelationService
from middleware.auth_middleware import token_required

bp = Blueprint('correlations', __name__)
correlation_service = CorrelationService()

@bp.route('/analyze', methods=['POST'])
@token_required
def analyze_correlations(current_user_id):
    """
    상관관계 분석 및 시각화
    """
    try:
        data = request.json
        file_id = data.get('file_id')
        target_column = data.get('target_column')
        features = data.get('features', [])
        
        if not all([file_id, target_column]):
            return jsonify({'error': 'file_id와 target_column이 필요합니다'}), 400
        
        result = correlation_service.analyze_correlations(
            file_id, target_column, features
        )
        
        return jsonify({
            'success': True,
            **result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/<file_id>', methods=['GET'])
@token_required
def get_correlations(current_user_id, file_id):
    """저장된 상관관계 분석 결과 조회"""
    result = correlation_service.get_correlations(file_id)
    if result:
        return jsonify({'success': True, 'result': result}), 200
    return jsonify({'error': '결과를 찾을 수 없습니다'}), 404


# ============================================
# app/api/predictions.py
# ============================================
from flask import Blueprint, request, jsonify
from services.prediction_service import PredictionService
from middleware.auth_middleware import token_required

bp = Blueprint('predictions', __name__)
prediction_service = PredictionService()

@bp.route('/forecast', methods=['POST'])
@token_required
def forecast(current_user_id):
    """
    수량 및 금액 예측
    """
    try:
        data = request.json
        file_id = data.get('file_id')
        target_columns = data.get('target_columns', ['수량', '금액'])
        forecast_days = data.get('forecast_days', 7)
        weights = data.get('weights', {})  # 상관관계에서 받은 가중치
        
        if not file_id:
            return jsonify({'error': 'file_id가 필요합니다'}), 400
        
        result = prediction_service.forecast(
            file_id, target_columns, weights, forecast_days
        )
        
        return jsonify({
            'success': True,
            **result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/<file_id>', methods=['GET'])
@token_required
def get_predictions(current_user_id, file_id):
    """저장된 예측 결과 조회"""
    result = prediction_service.get_predictions(file_id)
    if result:
        return jsonify({'success': True, 'result': result}), 200
    return jsonify({'error': '결과를 찾을 수 없습니다'}), 404


# ============================================
# app/api/solutions.py
# ============================================
from flask import Blueprint, request, jsonify
from services.llm_service import LLMService
from middleware.auth_middleware import token_required

bp = Blueprint('solutions', __name__)
llm_service = LLMService()

@bp.route('/generate', methods=['POST'])
@token_required
def generate_solution(current_user_id):
    """
    모든 분석 결과를 LLM에 전달하여 솔루션 생성
    """
    try:
        data = request.json
        file_id = data.get('file_id')
        include_visualizations = data.get('include_visualizations', True)
        include_correlations = data.get('include_correlations', True)
        include_predictions = data.get('include_predictions', True)
        
        if not file_id:
            return jsonify({'error': 'file_id가 필요합니다'}), 400
        
        solution = llm_service.generate_solution(
            file_id,
            include_visualizations=include_visualizations,
            include_correlations=include_correlations,
            include_predictions=include_predictions
        )
        
        return jsonify({
            'success': True,
            'solution': solution
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/<file_id>', methods=['GET'])
@token_required
def get_solution(current_user_id, file_id):
    """저장된 솔루션 조회"""
    result = llm_service.get_solution(file_id)
    if result:
        return jsonify({'success': True, 'solution': result}), 200
    return jsonify({'error': '솔루션을 찾을 수 없습니다'}), 404


# ============================================
# app/services/correlation_service.py
# ============================================
from services.base_service import BaseService
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
from io import BytesIO

class CorrelationService(BaseService):
    """상관관계 분석 서비스"""
    
    def analyze_correlations(self, file_id: str, target_column: str, features: list) -> dict:
        """
        상관관계 분석 및 가중치 계산
        """
        # 1. 데이터 로드
        df = self.load_data_from_mongodb(file_id)
        
        # 2. 상관계수 계산
        correlation_matrix = {}
        correlations = {}
        
        for feature in features:
            if feature in df.columns:
                corr = df[[target_column, feature]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations[feature] = float(corr)
        
        # 3. 가중치 계산 (상관계수 기반)
        weights = self.calculate_weights(correlations)
        
        # 4. 시각화 생성
        chart = self.create_correlation_chart(correlations, target_column)
        
        # 5. 결과 저장
        self.save_correlations(file_id, target_column, correlations, weights, chart)
        
        return {
            'correlation_matrix': {target_column: correlations},
            'top_correlations': sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5],
            'chart': chart,
            'weights': weights
        }
    
    def calculate_weights(self, correlations: dict) -> dict:
        """상관계수 기반 가중치 계산"""
        # 절댓값 사용
        abs_correlations = {k: abs(v) for k, v in correlations.items()}
        
        # 정규화 (합이 1이 되도록)
        total = sum(abs_correlations.values())
        if total > 0:
            weights = {k: v/total for k, v in abs_correlations.items()}
        else:
            weights = {k: 1/len(abs_correlations) for k in abs_correlations.keys()}
        
        return weights
    
    def create_correlation_chart(self, correlations: dict, target: str) -> str:
        """상관관계 차트 생성"""
        features = list(correlations.keys())
        values = list(correlations.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=features,
                y=values,
                marker_color=['red' if v < 0 else 'blue' for v in values]
            )
        ])
        fig.update_layout(
            title=f'{target}와의 상관관계',
            xaxis_title='피처',
            yaxis_title='상관계수',
            height=400
        )
        
        # Base64로 변환
        img_buffer = BytesIO()
        fig.write_image(img_buffer, format='png')
        return base64.b64encode(img_buffer.getvalue()).decode()
    
    def save_correlations(self, file_id: str, target: str, correlations: dict, weights: dict, chart: str):
        """상관관계 분석 결과 저장"""
        doc = {
            'correlation_id': f"corr_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'file_id': file_id,
            'user_id': self.mongo.files.find_one({'file_id': file_id})['user_id'],
            'target_column': target,
            'correlation_matrix': correlations,
            'weights': weights,
            'chart': chart,
            'created_at': datetime.now()
        }
        self.mongo.db['correlations'].insert_one(doc)


# ============================================
# app/services/llm_service.py
# ============================================
from services.base_service import BaseService
from services.visualization_service import VisualizationService
from services.correlation_service import CorrelationService
from services.prediction_service import PredictionService

class LLMService(BaseService):
    """LLM 솔루션 생성 서비스"""
    
    def __init__(self):
        super().__init__()
        self.viz_service = VisualizationService()
        self.corr_service = CorrelationService()
        self.pred_service = PredictionService()
        # LLM 클라이언트 초기화
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except:
            self.llm_client = None
    
    def generate_solution(self, file_id: str, 
                         include_visualizations: bool = True,
                         include_correlations: bool = True,
                         include_predictions: bool = True) -> dict:
        """
        모든 분석 결과를 수집하여 LLM 솔루션 생성
        """
        # 1. 모든 분석 결과 수집
        context = {}
        
        if include_visualizations:
            viz = self.viz_service.get_visualizations(file_id)
            if viz:
                context['visualizations'] = {
                    'statistics': viz.get('statistics', {}),
                    'summary': '시각화 데이터가 있습니다'
                }
        
        if include_correlations:
            corr = self.corr_service.get_correlations(file_id)
            if corr:
                context['correlations'] = {
                    'top_correlations': corr.get('correlation_matrix', {}),
                    'weights': corr.get('weights', {})
                }
        
        if include_predictions:
            pred = self.pred_service.get_predictions(file_id)
            if pred:
                context['predictions'] = {
                    'quantity': pred.get('predictions', {}).get('quantity', {}),
                    'amount': pred.get('predictions', {}).get('amount', {}),
                    'metrics': pred.get('model_metrics', {})
                }
        
        # 2. LLM 프롬프트 생성
        prompt = self._create_solution_prompt(context)
        
        # 3. LLM 호출
        if self.llm_client:
            solution_text = self._call_llm(prompt)
        else:
            solution_text = self._mock_solution(context)
        
        # 4. 솔루션 구조화
        solution = self._parse_solution(solution_text, context)
        
        # 5. 저장
        self.save_solution(file_id, solution)
        
        return solution
    
    def _create_solution_prompt(self, context: dict) -> str:
        """LLM 프롬프트 생성"""
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
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """LLM API 호출"""
        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 데이터 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def _parse_solution(self, solution_text: str, context: dict) -> dict:
        """LLM 응답을 구조화"""
        # 간단한 파싱 (실제로는 더 정교하게)
        return {
            'summary': solution_text.split('**요약**:')[1].split('**')[0].strip() if '**요약**:' in solution_text else solution_text[:200],
            'insights': self._extract_list(solution_text, '인사이트'),
            'recommendations': self._extract_list(solution_text, '권장사항'),
            'action_items': self._extract_action_items(solution_text)
        }
    
    def _extract_list(self, text: str, keyword: str) -> list:
        """리스트 추출"""
        # 간단한 구현
        items = []
        if keyword in text:
            section = text.split(keyword)[1].split('**')[0]
            for line in section.split('\n'):
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    items.append(line.strip().lstrip('-•').strip())
        return items[:5]  # 최대 5개
    
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
    
    def _mock_solution(self, context: dict) -> str:
        """모의 솔루션 (API 없을 때)"""
        return f"""
**요약**: 분석 결과를 종합하면, 데이터에서 몇 가지 중요한 패턴이 발견되었습니다.

**주요 인사이트**:
- 상관관계 분석 결과 주요 변수들이 확인되었습니다
- 예측 모델의 정확도가 양호한 수준입니다
- 추가 개선 여지가 있습니다

**권장사항**:
- 예측 결과를 바탕으로 재고 관리 최적화
- 주요 변수 모니터링 강화
- 정기적인 모델 재학습

**실행 항목**:
- [high] 재고 수준 조정
- [medium] 마케팅 전략 수정
"""
    
    def save_solution(self, file_id: str, solution: dict):
        """솔루션 저장"""
        doc = {
            'solution_id': f"sol_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'file_id': file_id,
            'user_id': self.mongo.files.find_one({'file_id': file_id})['user_id'],
            'solution': solution,
            'created_at': datetime.now()
        }
        self.mongo.db['solutions'].insert_one(doc)
    
    def get_solution(self, file_id: str) -> dict:
        """저장된 솔루션 조회"""
        solution = self.mongo.db['solutions'].find_one(
            {'file_id': file_id},
            sort=[('created_at', -1)]
        )
        return solution.get('solution', {}) if solution else None


# ============================================
# run.py
# ============================================
from app import create_app
import os
from dotenv import load_dotenv

load_dotenv()

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, port=port, host='0.0.0.0')

