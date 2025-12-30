"""
자동화 피처 엔지니어링 및 가중치 생성 파이프라인
- 어떤 CSV를 넣어도 자동으로 컬럼 감지, 피처 생성, 가중치 계산
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AutoFeaturePipeline:
    """자동화 피처 엔지니어링 파이프라인"""
    
    def __init__(self, target_column: Optional[str] = None):
        """
        Args:
            target_column: 예측할 타겟 컬럼명 (None이면 자동 감지)
        """
        self.target_column = target_column
        self.encoders = {}
        self.scaler = MinMaxScaler()
        self.feature_config = {}
        self.correlation_weights = {}
        
    def detect_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        컬럼 자동 감지 및 분류
        
        Returns:
            {
                'date_columns': [...],
                'categorical_columns': [...],
                'numeric_columns': [...],
                'target_candidates': [...]
            }
        """
        result = {
            'date_columns': [],
            'categorical_columns': [],
            'numeric_columns': [],
            'target_candidates': []
        }
        
        for col in df.columns:
            # 날짜 컬럼 감지
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(10))
                    result['date_columns'].append(col)
                    continue
                except:
                    pass
            
            # 범주형 vs 수치형
            if df[col].dtype == 'object' or df[col].nunique() < df.shape[0] * 0.1:
                result['categorical_columns'].append(col)
            else:
                result['numeric_columns'].append(col)
                
                # 타겟 후보 (수치형 중에서)
                if col.lower() in ['수량', 'quantity', 'qty', 'sales', '판매량', 'amount', '금액', 'price']:
                    result['target_candidates'].append(col)
        
        return result
    
    def auto_feature_engineering(self, df: pd.DataFrame, 
                                 group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        자동 피처 엔지니어링
        
        Args:
            df: 입력 데이터프레임
            group_by: 그룹화할 컬럼 (예: ['상품명', '지역'])
        
        Returns:
            피처가 추가된 데이터프레임
        """
        df = df.copy()
        column_info = self.detect_columns(df)
        
        # 1. 날짜 피처 생성
        for date_col in column_info['date_columns']:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[f'{date_col}_month'] = df[date_col].dt.month
            df[f'{date_col}_day_of_week'] = df[date_col].dt.dayofweek
            df[f'{date_col}_is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
            df[f'{date_col}_day'] = df[date_col].dt.day
        
        # 2. 타겟 컬럼 자동 감지
        if not self.target_column:
            if column_info['target_candidates']:
                self.target_column = column_info['target_candidates'][0]
            elif column_info['numeric_columns']:
                # 가장 마지막 수치형 컬럼을 타겟으로
                self.target_column = column_info['numeric_columns'][-1]
            else:
                raise ValueError("타겟 컬럼을 찾을 수 없습니다.")
        
        print(f"✅ 타겟 컬럼: {self.target_column}")
        
        # 3. 시계열 피처 생성 (group_by 기준)
        if group_by and self.target_column:
            for lag in [1, 7, 14]:
                df[f'{self.target_column}_lag_{lag}'] = df.groupby(group_by)[self.target_column].shift(lag)
            
            # 이동평균
            for window in [3, 7]:
                df[f'{self.target_column}_MA{window}'] = df.groupby(group_by)[self.target_column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                ).shift(1)
            
            # 변화량
            if f'{self.target_column}_lag_1' in df.columns:
                df[f'{self.target_column}_change'] = df[self.target_column] - df[f'{self.target_column}_lag_1']
        
        # 4. 범주형 인코딩
        for cat_col in column_info['categorical_columns']:
            if cat_col not in df.columns:
                continue
            le = LabelEncoder()
            df[f'{cat_col}_encoded'] = le.fit_transform(df[cat_col].astype(str))
            self.encoders[cat_col] = le
        
        # 5. 수치형 정규화 (선택적)
        numeric_cols = [col for col in column_info['numeric_columns'] 
                       if col != self.target_column and col in df.columns]
        if numeric_cols:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def calculate_correlation_weights(self, df: pd.DataFrame, 
                                     target_col: str,
                                     feature_cols: List[str]) -> Dict[str, float]:
        """
        상관계수 기반 가중치 계산
        
        Returns:
            {feature_name: correlation_weight}
        """
        weights = {}
        
        for feat in feature_cols:
            if feat in df.columns:
                corr = df[[target_col, feat]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    weights[feat] = abs(corr)  # 절댓값 사용
        
        # 정규화 (합이 1이 되도록)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def create_weighted_score(self, df: pd.DataFrame, 
                             weights: Dict[str, float]) -> pd.Series:
        """
        가중합 점수 생성
        """
        weighted_score = pd.Series(0.0, index=df.index)
        
        for feat, weight in weights.items():
            if feat in df.columns:
                weighted_score += df[feat] * weight
        
        return weighted_score
    
    def train_model(self, df: pd.DataFrame, 
                   feature_columns: List[str],
                   target_column: str,
                   test_size: float = 0.2) -> Tuple[RandomForestRegressor, Dict]:
        """
        모델 학습 및 평가
        
        Returns:
            (model, metrics)
        """
        # 결측치 제거
        df_clean = df[feature_columns + [target_column]].dropna()
        
        if len(df_clean) < 50:
            raise ValueError(f"데이터가 부족합니다. (최소 50개 필요, 현재 {len(df_clean)}개)")
        
        X = df_clean[feature_columns]
        y = df_clean[target_column]
        
        # 시계열 데이터이므로 순차적 분할
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # 모델 학습
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = (1 - mae / y_test.mean()) * 100 if y_test.mean() != 0 else 0
        
        metrics = {
            'mae': float(mae),
            'r2': float(r2),
            'accuracy': float(accuracy),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return model, metrics
    
    def process_csv(self, csv_path: str, 
                   group_by: Optional[List[str]] = None,
                   save_config: bool = True) -> Dict:
        """
        CSV 파일 전체 처리 파이프라인
        
        Returns:
            {
                'data': processed_df,
                'model': trained_model,
                'metrics': metrics,
                'weights': correlation_weights,
                'config': feature_config
            }
        """
        print(f"📁 CSV 파일 로드: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Shape: {df.shape}")
        
        # 컬럼 감지
        column_info = self.detect_columns(df)
        print(f"\n📊 컬럼 분류:")
        print(f"   - 날짜: {column_info['date_columns']}")
        print(f"   - 범주형: {column_info['categorical_columns']}")
        print(f"   - 수치형: {column_info['numeric_columns']}")
        print(f"   - 타겟 후보: {column_info['target_candidates']}")
        
        # 피처 엔지니어링
        print(f"\n🔧 피처 엔지니어링 중...")
        df_processed = self.auto_feature_engineering(df, group_by=group_by)
        print(f"   처리 후 Shape: {df_processed.shape}")
        
        # 피처 컬럼 선택
        feature_cols = []
        for col in df_processed.columns:
            if col != self.target_column and not col.endswith('_encoded'):
                if df_processed[col].dtype in ['int64', 'float64']:
                    feature_cols.append(col)
        
        # 인코딩된 컬럼도 추가
        encoded_cols = [col for col in df_processed.columns if col.endswith('_encoded')]
        feature_cols.extend(encoded_cols)
        
        print(f"\n📋 선택된 피처 ({len(feature_cols)}개):")
        print(f"   {feature_cols[:10]}...")
        
        # 상관계수 가중치 계산
        print(f"\n⚖️ 가중치 계산 중...")
        correlation_weights = self.calculate_correlation_weights(
            df_processed, self.target_column, feature_cols
        )
        self.correlation_weights = correlation_weights
        
        # 가중합 점수 생성
        df_processed['weighted_score'] = self.create_weighted_score(
            df_processed, correlation_weights
        )
        feature_cols.append('weighted_score')
        
        # 모델 학습
        print(f"\n🤖 모델 학습 중...")
        model, metrics = self.train_model(
            df_processed, feature_cols, self.target_column
        )
        
        print(f"\n✅ 완료!")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   R²: {metrics['r2']:.4f}")
        print(f"   정확도: {metrics['accuracy']:.2f}%")
        
        # 설정 저장
        self.feature_config = {
            'target_column': self.target_column,
            'feature_columns': feature_cols,
            'group_by': group_by,
            'column_info': column_info,
            'correlation_weights': correlation_weights,
            'metrics': metrics,
            'created_at': datetime.now().isoformat()
        }
        
        if save_config:
            config_path = csv_path.replace('.csv', '_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_config, f, ensure_ascii=False, indent=2)
            print(f"\n💾 설정 저장: {config_path}")
        
        return {
            'data': df_processed,
            'model': model,
            'metrics': metrics,
            'weights': correlation_weights,
            'config': self.feature_config
        }


# 사용 예시
if __name__ == "__main__":
    # 예시 1: 기본 사용
    pipeline = AutoFeaturePipeline()
    
    # CSV 처리
    result = pipeline.process_csv(
        csv_path='data/blinkit_with_weather.csv',
        group_by=['상품명', '지역']  # 지역별, 상품별 그룹화
    )
    
    print("\n" + "="*60)
    print("📊 결과 요약")
    print("="*60)
    print(f"처리된 데이터: {result['data'].shape}")
    print(f"모델 정확도: {result['metrics']['accuracy']:.2f}%")
    print(f"\n상위 5개 피처 가중치:")
    sorted_weights = sorted(result['weights'].items(), key=lambda x: x[1], reverse=True)
    for feat, weight in sorted_weights[:5]:
        print(f"  {feat}: {weight:.4f}")

