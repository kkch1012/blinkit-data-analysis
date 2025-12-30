"""
MongoDB 스키마 설계 및 웹 서비스용 데이터 모델
"""

from pymongo import MongoClient
from datetime import datetime
from typing import Dict, List, Optional
import json


class MongoDBService:
    """MongoDB 기반 데이터 서비스"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", db_name: str = "blinkit_analytics"):
        """
        Args:
            connection_string: MongoDB 연결 문자열
            db_name: 데이터베이스 이름
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        
        # 컬렉션 초기화
        self.files = self.db['files']  # 파일 메타데이터
        self.csv_contents = self.db['csv_contents']  # 실제 CSV 데이터
        self.analysis_results = self.db['analysis_results']  # 분석 결과
        self.feature_weights = self.db['feature_weights']  # 피처 가중치
        self.user_suggestions = self.db['user_suggestions']  # 사용자 제안
    
    def upload_csv(self, user_id: str, file_path: str, 
                   file_name: str, file_size: int) -> Dict:
        """
        CSV 파일 업로드 및 파싱
        
        Args:
            user_id: 사용자 ID
            file_path: 파일 경로
            file_name: 파일명
            file_size: 파일 크기
        
        Returns:
            {
                'file_id': str,
                'columns': List[str],
                'row_count': int,
                'suggestions': List[str]
            }
        """
        import pandas as pd
        
        # CSV 읽기
        df = pd.read_csv(file_path)
        
        # 파일 메타데이터 저장
        file_doc = {
            'file_id': f"file_{datetime.now().strftime('%Y%m%d%H%M%S')}_{user_id}",
            'user_id': user_id,
            'file_name': file_name,
            'file_size': file_size,
            'columns': df.columns.tolist(),
            'column_types': {col: str(df[col].dtype) for col in df.columns},
            'row_count': len(df),
            'uploaded_at': datetime.now(),
            'status': 'uploaded'
        }
        
        file_id = self.files.insert_one(file_doc).inserted_id
        file_doc['_id'] = file_id
        
        # CSV 데이터 대량 삽입
        csv_data = df.to_dict('records')
        csv_docs = []
        
        for idx, row in enumerate(csv_data):
            doc = {
                'file_id': file_doc['file_id'],
                'user_id': user_id,
                'row_index': idx,
                'data': row,
                'created_at': datetime.now()
            }
            csv_docs.append(doc)
            
            # 배치 삽입 (1000개씩)
            if len(csv_docs) >= 1000:
                self.csv_contents.insert_many(csv_docs)
                csv_docs = []
        
        # 남은 데이터 삽입
        if csv_docs:
            self.csv_contents.insert_many(csv_docs)
        
        # 자동 분석 및 제안 생성
        suggestions = self.generate_suggestions(file_doc)
        
        return {
            'file_id': file_doc['file_id'],
            'columns': df.columns.tolist(),
            'row_count': len(df),
            'suggestions': suggestions
        }
    
    def generate_suggestions(self, file_doc: Dict) -> List[str]:
        """
        파일 컬럼을 보고 자동 제안 생성
        
        Returns:
            제안 메시지 리스트
        """
        suggestions = []
        columns = file_doc['columns']
        column_types = file_doc['column_types']
        
        # 금액/수량 컬럼 감지
        amount_keywords = ['금액', 'amount', 'price', '가격', '매출', 'revenue']
        quantity_keywords = ['수량', 'quantity', 'qty', '판매량', 'sales']
        date_keywords = ['날짜', 'date', '일자', 'time', '시간']
        
        has_amount = any(kw in col.lower() for col in columns for kw in amount_keywords)
        has_quantity = any(kw in col.lower() for col in columns for kw in quantity_keywords)
        has_date = any(kw in col.lower() for col in columns for kw in date_keywords)
        
        if has_amount:
            suggestions.append("💰 '금액' 컬럼이 있네요! 합계/평균을 구해드릴까요?")
        
        if has_quantity:
            suggestions.append("📦 '수량' 컬럼이 있네요! 총 판매량을 계산해드릴까요?")
        
        if has_date:
            suggestions.append("📅 날짜 컬럼이 있네요! 시계열 분석을 진행할까요?")
            suggestions.append("📈 주간/월간 트렌드 분석을 해드릴 수 있습니다!")
        
        # 범주형 컬럼 감지
        categorical_cols = [col for col, dtype in column_types.items() 
                           if dtype == 'object' or col in ['상품명', '카테고리', '지역']]
        if categorical_cols:
            suggestions.append(f"🏷️ 범주형 컬럼({', '.join(categorical_cols[:3])})이 있네요! 그룹별 집계를 할까요?")
        
        # 제안 저장
        suggestion_doc = {
            'file_id': file_doc['file_id'],
            'user_id': file_doc['user_id'],
            'suggestions': suggestions,
            'created_at': datetime.now()
        }
        self.user_suggestions.insert_one(suggestion_doc)
        
        return suggestions
    
    def save_analysis_result(self, file_id: str, user_id: str,
                           analysis_type: str, result: Dict) -> str:
        """
        분석 결과 저장
        
        Args:
            file_id: 파일 ID
            user_id: 사용자 ID
            analysis_type: 분석 유형 ('correlation', 'model', 'trend' 등)
            result: 분석 결과
        
        Returns:
            analysis_id
        """
        analysis_doc = {
            'analysis_id': f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'file_id': file_id,
            'user_id': user_id,
            'analysis_type': analysis_type,
            'result': result,
            'created_at': datetime.now()
        }
        
        analysis_id = self.analysis_results.insert_one(analysis_doc).inserted_id
        return str(analysis_id)
    
    def save_feature_weights(self, file_id: str, user_id: str,
                            weights: Dict[str, float],
                            model_metrics: Dict) -> str:
        """
        피처 가중치 저장
        
        Args:
            file_id: 파일 ID
            user_id: 사용자 ID
            weights: {feature_name: weight}
            model_metrics: 모델 성능 지표
        
        Returns:
            weight_id
        """
        weight_doc = {
            'weight_id': f"weight_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'file_id': file_id,
            'user_id': user_id,
            'weights': weights,
            'model_metrics': model_metrics,
            'created_at': datetime.now()
        }
        
        weight_id = self.feature_weights.insert_one(weight_doc).inserted_id
        return str(weight_id)
    
    def get_file_data(self, file_id: str, limit: int = 100) -> List[Dict]:
        """
        파일 데이터 조회
        
        Args:
            file_id: 파일 ID
            limit: 조회할 행 수
        
        Returns:
            데이터 리스트
        """
        cursor = self.csv_contents.find(
            {'file_id': file_id}
        ).sort('row_index', 1).limit(limit)
        
        return [doc['data'] for doc in cursor]
    
    def get_user_files(self, user_id: str) -> List[Dict]:
        """
        사용자의 모든 파일 목록 조회
        """
        files = list(self.files.find({'user_id': user_id}).sort('uploaded_at', -1))
        return files
    
    def get_suggestions(self, file_id: str) -> List[str]:
        """
        파일에 대한 제안 조회
        """
        suggestion = self.user_suggestions.find_one({'file_id': file_id})
        return suggestion['suggestions'] if suggestion else []


# MongoDB 스키마 문서화
SCHEMA_DOCUMENTATION = """
# MongoDB 스키마 설계

## 1. files 컬렉션 (파일 메타데이터)
```json
{
  "_id": ObjectId,
  "file_id": "file_20241229120000_user123",
  "user_id": "user123",
  "file_name": "blinkit_data.csv",
  "file_size": 1024000,
  "columns": ["주문날짜", "상품명", "수량", "금액"],
  "column_types": {
    "주문날짜": "object",
    "상품명": "object",
    "수량": "int64",
    "금액": "float64"
  },
  "row_count": 5000,
  "uploaded_at": ISODate("2024-12-29T12:00:00Z"),
  "status": "uploaded"
}
```

## 2. csv_contents 컬렉션 (실제 데이터)
```json
{
  "_id": ObjectId,
  "file_id": "file_20241229120000_user123",
  "user_id": "user123",
  "row_index": 0,
  "data": {
    "주문날짜": "2024-07-17",
    "상품명": "Pet Treats",
    "수량": 3,
    "금액": 1551.09
  },
  "created_at": ISODate("2024-12-29T12:00:00Z")
}
```

## 3. analysis_results 컬렉션 (분석 결과)
```json
{
  "_id": ObjectId,
  "analysis_id": "analysis_20241229120000",
  "file_id": "file_20241229120000_user123",
  "user_id": "user123",
  "analysis_type": "correlation",
  "result": {
    "correlation_matrix": {...},
    "top_correlations": [...]
  },
  "created_at": ISODate("2024-12-29T12:00:00Z")
}
```

## 4. feature_weights 컬렉션 (피처 가중치)
```json
{
  "_id": ObjectId,
  "weight_id": "weight_20241229120000",
  "file_id": "file_20241229120000_user123",
  "user_id": "user123",
  "weights": {
    "수량_lag_1": 0.25,
    "temp_max": 0.15,
    "spend": 0.20,
    ...
  },
  "model_metrics": {
    "mae": 1.23,
    "r2": 0.65,
    "accuracy": 72.5
  },
  "created_at": ISODate("2024-12-29T12:00:00Z")
}
```

## 5. user_suggestions 컬렉션 (사용자 제안)
```json
{
  "_id": ObjectId,
  "file_id": "file_20241229120000_user123",
  "user_id": "user123",
  "suggestions": [
    "💰 '금액' 컬럼이 있네요! 합계/평균을 구해드릴까요?",
    "📦 '수량' 컬럼이 있네요! 총 판매량을 계산해드릴까요?",
    "📅 날짜 컬럼이 있네요! 시계열 분석을 진행할까요?"
  ],
  "created_at": ISODate("2024-12-29T12:00:00Z")
}
```

## 인덱스 설계
```javascript
// files 컬렉션
db.files.createIndex({ "user_id": 1, "uploaded_at": -1 })
db.files.createIndex({ "file_id": 1 })

// csv_contents 컬렉션
db.csv_contents.createIndex({ "file_id": 1, "row_index": 1 })
db.csv_contents.createIndex({ "user_id": 1 })

// analysis_results 컬렉션
db.analysis_results.createIndex({ "file_id": 1, "analysis_type": 1 })
db.analysis_results.createIndex({ "user_id": 1, "created_at": -1 })

// feature_weights 컬렉션
db.feature_weights.createIndex({ "file_id": 1 })
db.feature_weights.createIndex({ "user_id": 1 })
```
"""


# 사용 예시
if __name__ == "__main__":
    # MongoDB 서비스 초기화
    mongo_service = MongoDBService(
        connection_string="mongodb://localhost:27017/",
        db_name="blinkit_analytics"
    )
    
    # CSV 업로드 시뮬레이션
    result = mongo_service.upload_csv(
        user_id="user123",
        file_path="data/blinkit_with_weather.csv",
        file_name="blinkit_with_weather.csv",
        file_size=1024000
    )
    
    print("="*60)
    print("📁 파일 업로드 완료")
    print("="*60)
    print(f"File ID: {result['file_id']}")
    print(f"컬럼 수: {len(result['columns'])}")
    print(f"행 수: {result['row_count']}")
    print(f"\n💡 자동 제안:")
    for suggestion in result['suggestions']:
        print(f"  - {suggestion}")
    
    # 피처 가중치 저장 예시
    weights = {
        '수량_lag_1': 0.25,
        'temp_max': 0.15,
        'spend': 0.20,
        'rainfall': 0.10
    }
    
    metrics = {
        'mae': 1.23,
        'r2': 0.65,
        'accuracy': 72.5
    }
    
    weight_id = mongo_service.save_feature_weights(
        file_id=result['file_id'],
        user_id="user123",
        weights=weights,
        model_metrics=metrics
    )
    
    print(f"\n💾 가중치 저장 완료: {weight_id}")

