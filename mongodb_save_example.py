"""
MongoDB 저장 예시: 컬렉션 구조에 맞게 데이터 저장하는 방법
"""

from pymongo import MongoClient
from datetime import datetime
import pandas as pd


class MongoDBDataSaver:
    """MongoDB에 데이터를 저장하는 클래스"""
    
    def __init__(self, connection_string="mongodb://localhost:27017/", db_name="blinkit_analytics"):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        
        # 컬렉션 초기화 (미리 만들어둠)
        self.files = self.db['files']
        self.csv_contents = self.db['csv_contents']
        self.user_suggestions = self.db['user_suggestions']
        self.analysis_results = self.db['analysis_results']
        self.feature_weights = self.db['feature_weights']
    
    def save_csv_file(self, user_id: str, csv_path: str, file_name: str):
        """
        CSV 파일을 MongoDB에 저장
        - 컬렉션 구조에 맞게 저장
        - CSV마다 컬럼명이 달라도 OK (data 필드에 JSON으로 저장)
        """
        # 1. CSV 읽기
        df = pd.read_csv(csv_path)
        
        # 2. file_id 생성
        file_id = f"file_{datetime.now().strftime('%Y%m%d%H%M%S')}_{user_id}"
        
        # 3. files 컬렉션에 메타데이터 저장 (구조 고정)
        file_doc = {
            "file_id": file_id,                    # ✅ 고정 필드
            "user_id": user_id,                     # ✅ 고정 필드
            "file_name": file_name,                 # ✅ 고정 필드
            "file_size": len(df),                   # ✅ 고정 필드
            "columns": df.columns.tolist(),         # ✅ 고정 필드 (배열)
            "column_types": {                       # ✅ 고정 필드 (딕셔너리)
                col: str(df[col].dtype) for col in df.columns
            },
            "row_count": len(df),                   # ✅ 고정 필드
            "uploaded_at": datetime.now(),          # ✅ 고정 필드
            "status": "uploaded"                    # ✅ 고정 필드
        }
        
        self.files.insert_one(file_doc)
        print(f"✅ files 컬렉션에 저장 완료: {file_id}")
        
        # 4. csv_contents 컬렉션에 실제 데이터 저장 (구조 고정, 내용 유연)
        csv_docs = []
        
        for idx, row in df.iterrows():
            # ✅ 컬렉션 구조는 고정 (file_id, user_id, row_index, data, created_at)
            # ✅ 하지만 data 필드 안의 내용은 CSV마다 달라도 OK!
            doc = {
                "file_id": file_id,                 # ✅ 고정 필드
                "user_id": user_id,                 # ✅ 고정 필드
                "row_index": int(idx),              # ✅ 고정 필드
                "data": row.to_dict(),              # ✅ 유연한 필드 (CSV마다 다름)
                "created_at": datetime.now()        # ✅ 고정 필드
            }
            csv_docs.append(doc)
            
            # 배치 삽입 (1000개씩)
            if len(csv_docs) >= 1000:
                self.csv_contents.insert_many(csv_docs)
                csv_docs = []
        
        # 남은 데이터 삽입
        if csv_docs:
            self.csv_contents.insert_many(csv_docs)
        
        print(f"✅ csv_contents 컬렉션에 {len(df)}개 행 저장 완료")
        
        return file_id
    
    def save_suggestions(self, file_id: str, user_id: str, suggestions: list):
        """
        제안 저장 (구조 고정)
        """
        suggestion_doc = {
            "file_id": file_id,                     # ✅ 고정 필드
            "user_id": user_id,                     # ✅ 고정 필드
            "suggestions": suggestions,             # ✅ 고정 필드 (배열)
            "created_at": datetime.now()            # ✅ 고정 필드
        }
        
        self.user_suggestions.insert_one(suggestion_doc)
        print(f"✅ user_suggestions 컬렉션에 저장 완료")
    
    def save_analysis_result(self, file_id: str, user_id: str, metrics: dict):
        """
        분석 결과 저장 (구조 고정)
        """
        analysis_doc = {
            "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "file_id": file_id,                     # ✅ 고정 필드
            "user_id": user_id,                     # ✅ 고정 필드
            "analysis_type": "auto_feature_engineering",  # ✅ 고정 필드
            "result": {                             # ✅ 고정 필드 (딕셔너리)
                "metrics": metrics,
                "created_at": datetime.now().isoformat()
            },
            "created_at": datetime.now()            # ✅ 고정 필드
        }
        
        self.analysis_results.insert_one(analysis_doc)
        print(f"✅ analysis_results 컬렉션에 저장 완료")
    
    def save_feature_weights(self, file_id: str, user_id: str, weights: dict, metrics: dict):
        """
        피처 가중치 저장 (구조 고정)
        """
        weight_doc = {
            "weight_id": f"weight_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "file_id": file_id,                     # ✅ 고정 필드
            "user_id": user_id,                     # ✅ 고정 필드
            "weights": weights,                     # ✅ 고정 필드 (딕셔너리)
            "model_metrics": metrics,               # ✅ 고정 필드 (딕셔너리)
            "created_at": datetime.now()            # ✅ 고정 필드
        }
        
        self.feature_weights.insert_one(weight_doc)
        print(f"✅ feature_weights 컬렉션에 저장 완료")


# 사용 예시
if __name__ == "__main__":
    saver = MongoDBDataSaver()
    
    # 예시 1: 한글 컬럼명 CSV
    print("="*60)
    print("예시 1: 한글 컬럼명 CSV 저장")
    print("="*60)
    file_id1 = saver.save_csv_file(
        user_id="user123",
        csv_path="data/blinkit_with_weather.csv",
        file_name="blinkit_with_weather.csv"
    )
    
    # 제안 저장
    saver.save_suggestions(
        file_id=file_id1,
        user_id="user123",
        suggestions=[
            "💰 '금액' 컬럼이 있네요! 합계를 구해드릴까요?",
            "📦 '수량' 컬럼이 있네요! 총 판매량을 계산해드릴까요?"
        ]
    )
    
    # 분석 결과 저장
    saver.save_analysis_result(
        file_id=file_id1,
        user_id="user123",
        metrics={
            "mae": 1.23,
            "r2": 0.65,
            "accuracy": 72.5
        }
    )
    
    # 가중치 저장
    saver.save_feature_weights(
        file_id=file_id1,
        user_id="user123",
        weights={
            "수량_lag_1": 0.25,
            "temp_max": 0.15,
            "spend": 0.20
        },
        metrics={
            "mae": 1.23,
            "r2": 0.65,
            "accuracy": 72.5
        }
    )
    
    print("\n" + "="*60)
    print("✅ 모든 저장 완료!")
    print("="*60)
    
    # 조회 예시
    print("\n📊 저장된 데이터 조회:")
    
    # files 컬렉션 조회
    file_info = saver.files.find_one({"file_id": file_id1})
    print(f"\n파일 정보:")
    print(f"  - 파일명: {file_info['file_name']}")
    print(f"  - 컬럼: {file_info['columns']}")
    print(f"  - 행 수: {file_info['row_count']}")
    
    # csv_contents 컬렉션 조회 (첫 3개 행)
    print(f"\n실제 데이터 (첫 3개 행):")
    for doc in saver.csv_contents.find({"file_id": file_id1}).limit(3):
        print(f"  행 {doc['row_index']}: {doc['data']}")


"""
핵심 정리:

1. 컬렉션 구조는 고정 (코드에서 정의)
   - files: file_id, user_id, file_name, columns, ...
   - csv_contents: file_id, user_id, row_index, data, ...

2. 하지만 data 필드 안의 내용은 유연 (CSV마다 다름)
   - CSV 1: {"주문날짜": "...", "상품명": "..."}
   - CSV 2: {"date": "...", "product": "..."}
   - 모두 같은 구조로 저장 가능!

3. 코드에서 저장할 때:
   - ✅ 컬렉션의 고정 필드는 항상 포함
   - ✅ data 필드에는 CSV의 모든 컬럼을 JSON으로 저장
   - ✅ CSV마다 컬럼명이 달라도 문제없음
"""

