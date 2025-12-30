"""
수량 및 금액 예측 서비스
"""

from auto_feature_pipeline import AutoFeaturePipeline
from mongodb_schema import MongoDBService
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class PredictionService:
    """수량 및 금액 예측 서비스"""
    
    def __init__(self):
        self.mongo = MongoDBService()
        self.pipeline_quantity = AutoFeaturePipeline()
        self.pipeline_amount = AutoFeaturePipeline()
    
    def load_data_from_mongodb(self, file_id: str) -> pd.DataFrame:
        """
        MongoDB에서 데이터 로드
        """
        # csv_contents에서 데이터 조회
        cursor = self.mongo.csv_contents.find({"file_id": file_id})
        
        # DataFrame으로 변환
        data_list = []
        for doc in cursor:
            row = doc['data'].copy()
            row['row_index'] = doc['row_index']
            data_list.append(row)
        
        df = pd.DataFrame(data_list)
        return df
    
    def predict_quantity_and_amount(self, file_id: str, forecast_days: int = 7):
        """
        수량과 금액을 동시에 예측
        
        Returns:
            {
                'quantity': {'dates': [...], 'predicted': [...], 'avg': float},
                'amount': {'dates': [...], 'predicted': [...], 'avg': float},
                'metrics': {
                    'quantity': {...},
                    'amount': {...}
                }
            }
        """
        print(f"📊 예측 시작: file_id={file_id}, forecast_days={forecast_days}")
        
        # 1. 데이터 로드
        df = self.load_data_from_mongodb(file_id)
        print(f"   데이터 로드 완료: {len(df)}행")
        
        # 2. 컬럼명 확인 및 타겟 설정
        # 수량 컬럼 찾기
        quantity_col = None
        for col in df.columns:
            if any(kw in col.lower() for kw in ['수량', 'quantity', 'qty', 'sales']):
                quantity_col = col
                break
        
        # 금액 컬럼 찾기
        amount_col = None
        for col in df.columns:
            if any(kw in col.lower() for kw in ['금액', 'amount', 'price', '매출', 'revenue']):
                amount_col = col
                break
        
        if not quantity_col or not amount_col:
            raise ValueError(f"수량 또는 금액 컬럼을 찾을 수 없습니다. 컬럼: {df.columns.tolist()}")
        
        print(f"   타겟 컬럼: 수량={quantity_col}, 금액={amount_col}")
        
        # 3. 수량 예측
        print(f"\n🔹 수량 예측 모델 학습 중...")
        self.pipeline_quantity.target_column = quantity_col
        result_quantity = self.pipeline_quantity.process_csv(
            csv_path=None,  # 이미 DataFrame이 있으므로
            group_by=None,
            save_config=False
        )
        
        # DataFrame 직접 처리
        df_processed = self.pipeline_quantity.auto_feature_engineering(df)
        
        # 모델 학습
        feature_cols = [col for col in df_processed.columns 
                       if col != quantity_col and df_processed[col].dtype in ['int64', 'float64']]
        model_q, metrics_q = self.pipeline_quantity.train_model(
            df_processed, feature_cols, quantity_col
        )
        
        # 4. 금액 예측
        print(f"\n🔹 금액 예측 모델 학습 중...")
        self.pipeline_amount.target_column = amount_col
        df_processed_amount = self.pipeline_amount.auto_feature_engineering(df)
        
        feature_cols_amount = [col for col in df_processed_amount.columns 
                              if col != amount_col and df_processed_amount[col].dtype in ['int64', 'float64']]
        model_a, metrics_a = self.pipeline_amount.train_model(
            df_processed_amount, feature_cols_amount, amount_col
        )
        
        # 5. 미래 예측
        print(f"\n🔮 미래 {forecast_days}일 예측 중...")
        quantity_forecast = self.forecast_future(
            model_q, df_processed, feature_cols, quantity_col, forecast_days
        )
        amount_forecast = self.forecast_future(
            model_a, df_processed_amount, feature_cols_amount, amount_col, forecast_days
        )
        
        # 6. 결과 저장
        predictions = {
            'quantity': quantity_forecast,
            'amount': amount_forecast,
            'metrics': {
                'quantity': metrics_q,
                'amount': metrics_a
            }
        }
        
        self.save_predictions(file_id, predictions)
        
        print(f"\n✅ 예측 완료!")
        print(f"   수량 평균: {quantity_forecast['avg']:.2f}개")
        print(f"   금액 평균: {amount_forecast['avg']:.2f}원")
        
        return predictions
    
    def forecast_future(self, model, df, feature_cols, target_col, days):
        """
        미래 예측
        """
        # 마지막 날짜 찾기
        date_cols = [col for col in df.columns if 'date' in col.lower() or '날짜' in col]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            last_date = df[date_cols[0]].max()
        else:
            last_date = datetime.now()
        
        # 예측 날짜 생성
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        # 마지막 데이터로부터 예측
        last_row = df.iloc[-1:].copy()
        predictions = []
        
        for i, date in enumerate(future_dates):
            # 피처 업데이트 (날짜 관련)
            if date_cols:
                last_row[date_cols[0]] = date
                last_row[f'{date_cols[0]}_month'] = date.month
                last_row[f'{date_cols[0]}_day_of_week'] = date.weekday()
                last_row[f'{date_cols[0]}_is_weekend'] = 1 if date.weekday() >= 5 else 0
            
            # 예측
            X = last_row[feature_cols].fillna(0)
            pred = model.predict(X)[0]
            predictions.append(pred)
        
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'predicted': [float(p) for p in predictions],
            'avg': float(np.mean(predictions))
        }
    
    def save_predictions(self, file_id: str, predictions: dict):
        """
        예측 결과를 MongoDB에 저장
        """
        prediction_doc = {
            'prediction_id': f"pred_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'file_id': file_id,
            'user_id': self.mongo.files.find_one({'file_id': file_id})['user_id'],
            'predictions': {
                'quantity': [
                    {'date': d, 'predicted': p, 'actual': None}
                    for d, p in zip(predictions['quantity']['dates'], 
                                   predictions['quantity']['predicted'])
                ],
                'amount': [
                    {'date': d, 'predicted': p, 'actual': None}
                    for d, p in zip(predictions['amount']['dates'], 
                                   predictions['amount']['predicted'])
                ]
            },
            'model_metrics': predictions['metrics'],
            'created_at': datetime.now()
        }
        
        # predictions 컬렉션에 저장 (없으면 자동 생성)
        self.mongo.db['predictions'].insert_one(prediction_doc)
        print(f"💾 예측 결과 저장 완료: {prediction_doc['prediction_id']}")
    
    def get_predictions(self, file_id: str):
        """
        저장된 예측 결과 조회
        """
        prediction = self.mongo.db['predictions'].find_one(
            {'file_id': file_id},
            sort=[('created_at', -1)]
        )
        return prediction

