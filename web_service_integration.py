"""
웹 서비스 통합: MongoDB + 자동화 파이프라인
"""

from auto_feature_pipeline import AutoFeaturePipeline
from mongodb_schema import MongoDBService
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 서비스 초기화
mongo_service = MongoDBService()
pipeline = AutoFeaturePipeline()


@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """
    CSV 파일 업로드 엔드포인트
    
    Request:
        - file: CSV 파일
        - user_id: 사용자 ID
    
    Response:
        {
            'file_id': str,
            'columns': List[str],
            'row_count': int,
            'suggestions': List[str]
        }
    """
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400
    
    file = request.files['file']
    user_id = request.form.get('user_id', 'anonymous')
    
    if file.filename == '':
        return jsonify({'error': '파일명이 없습니다'}), 400
    
    # 파일 저장
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    
    # MongoDB에 업로드
    file_size = os.path.getsize(filepath)
    result = mongo_service.upload_csv(
        user_id=user_id,
        file_path=filepath,
        file_name=filename,
        file_size=file_size
    )
    
    return jsonify(result), 200


@app.route('/api/analyze', methods=['POST'])
def analyze_csv():
    """
    CSV 자동 분석 및 피처 가중치 생성
    
    Request:
        {
            'file_id': str,
            'target_column': str (optional),
            'group_by': List[str] (optional)
        }
    
    Response:
        {
            'analysis_id': str,
            'metrics': Dict,
            'weights': Dict,
            'suggestions': List[str]
        }
    """
    data = request.json
    file_id = data.get('file_id')
    target_column = data.get('target_column')
    group_by = data.get('group_by', [])
    
    # 파일 정보 조회
    file_info = mongo_service.files.find_one({'file_id': file_id})
    if not file_info:
        return jsonify({'error': '파일을 찾을 수 없습니다'}), 404
    
    # 파일 경로 구성
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_info['file_name'])
    
    # 자동화 파이프라인 실행
    pipeline.target_column = target_column
    result = pipeline.process_csv(
        csv_path=filepath,
        group_by=group_by if group_by else None,
        save_config=True
    )
    
    # 결과 저장
    analysis_id = mongo_service.save_analysis_result(
        file_id=file_id,
        user_id=file_info['user_id'],
        analysis_type='auto_feature_engineering',
        result={
            'metrics': result['metrics'],
            'feature_count': len(result['config']['feature_columns'])
        }
    )
    
    # 가중치 저장
    weight_id = mongo_service.save_feature_weights(
        file_id=file_id,
        user_id=file_info['user_id'],
        weights=result['weights'],
        model_metrics=result['metrics']
    )
    
    # 제안 조회
    suggestions = mongo_service.get_suggestions(file_id)
    
    return jsonify({
        'analysis_id': analysis_id,
        'weight_id': weight_id,
        'metrics': result['metrics'],
        'weights': result['weights'],
        'suggestions': suggestions
    }), 200


@app.route('/api/files/<user_id>', methods=['GET'])
def get_user_files(user_id):
    """
    사용자의 파일 목록 조회
    """
    files = mongo_service.get_user_files(user_id)
    
    # ObjectId를 문자열로 변환
    for file in files:
        file['_id'] = str(file['_id'])
        file['uploaded_at'] = file['uploaded_at'].isoformat()
    
    return jsonify({'files': files}), 200


@app.route('/api/data/<file_id>', methods=['GET'])
def get_file_data(file_id):
    """
    파일 데이터 조회
    """
    limit = request.args.get('limit', 100, type=int)
    data = mongo_service.get_file_data(file_id, limit=limit)
    
    return jsonify({'data': data, 'count': len(data)}), 200


@app.route('/api/weights/<file_id>', methods=['GET'])
def get_feature_weights(file_id):
    """
    피처 가중치 조회
    """
    weight = mongo_service.feature_weights.find_one(
        {'file_id': file_id},
        sort=[('created_at', -1)]
    )
    
    if not weight:
        return jsonify({'error': '가중치를 찾을 수 없습니다'}), 404
    
    weight['_id'] = str(weight['_id'])
    weight['created_at'] = weight['created_at'].isoformat()
    
    return jsonify(weight), 200


@app.route('/api/suggestions/<file_id>', methods=['GET'])
def get_suggestions(file_id):
    """
    파일 제안 조회
    """
    suggestions = mongo_service.get_suggestions(file_id)
    return jsonify({'suggestions': suggestions}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)

