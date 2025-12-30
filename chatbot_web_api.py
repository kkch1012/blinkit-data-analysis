"""
챗봇 웹 API - Flask 기반
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from prediction_service import PredictionService
from visualization_service import VisualizationService
from llm_chatbot_service import LLMChatbotService
from mongodb_schema import MongoDBService
import os

app = Flask(__name__)
CORS(app)  # CORS 허용 (프론트엔드와 통신)

# 서비스 초기화
prediction_service = PredictionService()
viz_service = VisualizationService()
mongo_service = MongoDBService()

# LLM API 키는 환경변수에서 가져오기
LLM_API_KEY = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
LLM_PROVIDER = "openai" if os.getenv('OPENAI_API_KEY') else "anthropic"
llm_service = LLMChatbotService(api_key=LLM_API_KEY, provider=LLM_PROVIDER)


@app.route('/api/analyze-and-predict', methods=['POST'])
def analyze_and_predict():
    """
    CSV 분석 → 예측 → 시각화 → LLM 인사이트 생성
    
    Request:
        {
            "file_id": "file_001",
            "forecast_days": 7
        }
    
    Response:
        {
            "predictions": {...},
            "charts": {...},
            "insights": "..."
        }
    """
    try:
        data = request.json
        file_id = data.get('file_id')
        forecast_days = data.get('forecast_days', 7)
        
        if not file_id:
            return jsonify({'error': 'file_id가 필요합니다'}), 400
        
        print(f"📊 분석 시작: file_id={file_id}")
        
        # 1. 예측
        predictions = prediction_service.predict_quantity_and_amount(
            file_id, forecast_days
        )
        
        # 2. 시각화
        charts = viz_service.create_forecast_charts(predictions, file_id)
        
        # 3. LLM 인사이트
        insights = llm_service.generate_insights(
            predictions,
            predictions['metrics']
        )
        
        return jsonify({
            'success': True,
            'predictions': {
                'quantity': predictions['quantity'],
                'amount': predictions['amount'],
                'metrics': predictions['metrics']
            },
            'charts': charts,
            'insights': insights
        }), 200
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    챗봇 대화
    
    Request:
        {
            "file_id": "file_001",
            "user_id": "user123",
            "question": "이번 주 수량 예측이 어떻게 되나요?"
        }
    
    Response:
        {
            "answer": "...",
            "insights": [...]
        }
    """
    try:
        data = request.json
        file_id = data.get('file_id')
        user_id = data.get('user_id')
        question = data.get('question')
        
        if not all([file_id, user_id, question]):
            return jsonify({'error': 'file_id, user_id, question이 필요합니다'}), 400
        
        # 예측 결과 및 시각화 로드
        predictions_doc = prediction_service.get_predictions(file_id)
        visualizations_doc = viz_service.get_visualizations(file_id)
        chat_history = llm_service.get_chat_history(file_id)
        
        # 예측 데이터 변환
        predictions = None
        if predictions_doc:
            predictions = {
                'quantity': {
                    'avg': sum([p['predicted'] for p in predictions_doc['predictions']['quantity']]) / len(predictions_doc['predictions']['quantity']),
                    'dates': [p['date'] for p in predictions_doc['predictions']['quantity']],
                    'predicted': [p['predicted'] for p in predictions_doc['predictions']['quantity']]
                },
                'amount': {
                    'avg': sum([p['predicted'] for p in predictions_doc['predictions']['amount']]) / len(predictions_doc['predictions']['amount']),
                    'dates': [p['date'] for p in predictions_doc['predictions']['amount']],
                    'predicted': [p['predicted'] for p in predictions_doc['predictions']['amount']]
                }
            }
        
        # LLM 답변
        answer = llm_service.answer_question(
            question, file_id, predictions, visualizations_doc, chat_history
        )
        
        return jsonify({
            'success': True,
            'answer': answer
        }), 200
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat-history/<file_id>', methods=['GET'])
def get_chat_history(file_id):
    """
    대화 이력 조회
    """
    try:
        history = llm_service.get_chat_history(file_id)
        return jsonify({
            'success': True,
            'history': history
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions/<file_id>', methods=['GET'])
def get_predictions(file_id):
    """
    예측 결과 조회
    """
    try:
        predictions = prediction_service.get_predictions(file_id)
        if predictions:
            return jsonify({
                'success': True,
                'predictions': predictions
            }), 200
        else:
            return jsonify({'error': '예측 결과를 찾을 수 없습니다'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualizations/<file_id>', methods=['GET'])
def get_visualizations(file_id):
    """
    시각화 조회
    """
    try:
        visualizations = viz_service.get_visualizations(file_id)
        if visualizations:
            return jsonify({
                'success': True,
                'visualizations': visualizations
            }), 200
        else:
            return jsonify({'error': '시각화를 찾을 수 없습니다'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """헬스 체크"""
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    print("🚀 챗봇 웹 API 서버 시작")
    print("=" * 60)
    print("📡 엔드포인트:")
    print("  POST /api/analyze-and-predict - 분석 및 예측")
    print("  POST /api/chat - 챗봇 대화")
    print("  GET  /api/chat-history/<file_id> - 대화 이력")
    print("  GET  /api/predictions/<file_id> - 예측 결과")
    print("  GET  /api/visualizations/<file_id> - 시각화")
    print("=" * 60)
    
    app.run(debug=True, port=5000, host='0.0.0.0')

