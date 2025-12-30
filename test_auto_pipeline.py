"""
자동화 파이프라인 테스트 스크립트
"""

from auto_feature_pipeline import AutoFeaturePipeline
import pandas as pd

def test_auto_pipeline():
    """자동화 파이프라인 테스트"""
    
    print("="*60)
    print("🚀 자동화 파이프라인 테스트")
    print("="*60)
    
    # 파이프라인 초기화
    pipeline = AutoFeaturePipeline()
    
    # 테스트 1: 기상 데이터 포함 CSV
    print("\n[테스트 1] 기상 데이터 포함 CSV 처리")
    print("-" * 60)
    try:
        result1 = pipeline.process_csv(
            csv_path='data/blinkit_with_weather.csv',
            group_by=['상품명', '지역']
        )
        print(f"✅ 성공!")
        print(f"   정확도: {result1['metrics']['accuracy']:.2f}%")
        print(f"   MAE: {result1['metrics']['mae']:.4f}")
        print(f"   R²: {result1['metrics']['r2']:.4f}")
        print(f"\n   상위 5개 피처 가중치:")
        sorted_weights = sorted(result1['weights'].items(), 
                              key=lambda x: x[1], reverse=True)
        for feat, weight in sorted_weights[:5]:
            print(f"     {feat}: {weight:.4f}")
    except Exception as e:
        print(f"❌ 실패: {e}")
    
    # 테스트 2: 다른 CSV (주간 데이터)
    print("\n[테스트 2] 주간 데이터 CSV 처리")
    print("-" * 60)
    try:
        result2 = pipeline.process_csv(
            csv_path='data/blinkit_weekly_product_weather.csv',
            group_by=['상품명', '지역']
        )
        print(f"✅ 성공!")
        print(f"   정확도: {result2['metrics']['accuracy']:.2f}%")
        print(f"   MAE: {result2['metrics']['mae']:.4f}")
    except Exception as e:
        print(f"❌ 실패: {e}")
    
    # 테스트 3: 컬럼 감지 테스트
    print("\n[테스트 3] 컬럼 자동 감지 테스트")
    print("-" * 60)
    try:
        df = pd.read_csv('data/blinkit_with_weather.csv')
        column_info = pipeline.detect_columns(df)
        
        print(f"✅ 감지 완료!")
        print(f"   날짜 컬럼: {column_info['date_columns']}")
        print(f"   범주형 컬럼: {column_info['categorical_columns'][:5]}...")
        print(f"   수치형 컬럼: {column_info['numeric_columns']}")
        print(f"   타겟 후보: {column_info['target_candidates']}")
    except Exception as e:
        print(f"❌ 실패: {e}")
    
    print("\n" + "="*60)
    print("✅ 모든 테스트 완료!")
    print("="*60)


if __name__ == "__main__":
    test_auto_pipeline()

