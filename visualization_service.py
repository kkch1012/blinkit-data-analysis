"""
시각화 서비스 - 예측 결과를 차트로 생성
"""

import plotly.graph_objects as go
import plotly.express as px
import base64
from io import BytesIO
from mongodb_schema import MongoDBService
from datetime import datetime
import pandas as pd


class VisualizationService:
    """시각화 서비스"""
    
    def __init__(self):
        self.mongo = MongoDBService()
    
    def create_forecast_charts(self, predictions: dict, file_id: str):
        """
        예측 결과를 시각화하여 Base64 이미지로 반환
        """
        charts = {}
        chart_data = {}
        
        # 1. 수량 예측 차트
        fig_quantity = go.Figure()
        fig_quantity.add_trace(go.Scatter(
            x=predictions['quantity']['dates'],
            y=predictions['quantity']['predicted'],
            name='예측 수량',
            line=dict(color='#3498db', width=3),
            mode='lines+markers'
        ))
        fig_quantity.add_hline(
            y=predictions['quantity']['avg'],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"평균: {predictions['quantity']['avg']:.2f}개"
        )
        fig_quantity.update_layout(
            title='📦 수량 예측 (향후 7일)',
            xaxis_title='날짜',
            yaxis_title='수량 (개)',
            template='plotly_white',
            height=400
        )
        
        charts['quantity_forecast'] = self.fig_to_base64(fig_quantity)
        chart_data['quantity_forecast'] = predictions['quantity']
        
        # 2. 금액 예측 차트
        fig_amount = go.Figure()
        fig_amount.add_trace(go.Scatter(
            x=predictions['amount']['dates'],
            y=predictions['amount']['predicted'],
            name='예측 금액',
            line=dict(color='#2ecc71', width=3),
            mode='lines+markers',
            fill='tonexty'
        ))
        fig_amount.add_hline(
            y=predictions['amount']['avg'],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"평균: {predictions['amount']['avg']:,.0f}원"
        )
        fig_amount.update_layout(
            title='💰 금액 예측 (향후 7일)',
            xaxis_title='날짜',
            yaxis_title='금액 (원)',
            template='plotly_white',
            height=400
        )
        
        charts['amount_forecast'] = self.fig_to_base64(fig_amount)
        chart_data['amount_forecast'] = predictions['amount']
        
        # 3. 피처 중요도 차트 (수량 모델)
        if 'feature_importance' in predictions.get('metrics', {}).get('quantity', {}):
            fig_importance = self.create_feature_importance_chart(
                predictions['metrics']['quantity']['feature_importance']
            )
            charts['feature_importance'] = self.fig_to_base64(fig_importance)
        
        # 4. 성능 대시보드
        fig_performance = self.create_performance_dashboard(predictions['metrics'])
        charts['performance_dashboard'] = self.fig_to_base64(fig_performance)
        
        # 5. MongoDB에 저장
        self.save_visualizations(file_id, charts, chart_data)
        
        return charts
    
    def fig_to_base64(self, fig) -> str:
        """
        Plotly Figure를 Base64 이미지로 변환
        """
        img_buffer = BytesIO()
        fig.write_image(img_buffer, format='png', width=800, height=400)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        return img_base64
    
    def create_feature_importance_chart(self, feature_importance: dict):
        """
        피처 중요도 차트 생성
        """
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker_color='#e74c3c'
            )
        ])
        fig.update_layout(
            title='📊 피처 중요도',
            xaxis_title='중요도',
            yaxis_title='피처',
            template='plotly_white',
            height=400
        )
        return fig
    
    def create_performance_dashboard(self, metrics: dict):
        """
        성능 대시보드 생성
        """
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('수량 모델 성능', '금액 모델 성능', '정확도 비교', 'MAE 비교'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 수량 모델 게이지
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics['quantity']['accuracy'],
                title={'text': "수량 정확도 (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"}},
            ),
            row=1, col=1
        )
        
        # 금액 모델 게이지
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics['amount']['accuracy'],
                title={'text': "금액 정확도 (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"}},
            ),
            row=1, col=2
        )
        
        # 정확도 비교
        fig.add_trace(
            go.Bar(x=['수량', '금액'],
                   y=[metrics['quantity']['accuracy'], metrics['amount']['accuracy']],
                   marker_color=['#3498db', '#2ecc71']),
            row=2, col=1
        )
        
        # MAE 비교
        fig.add_trace(
            go.Bar(x=['수량', '금액'],
                   y=[metrics['quantity']['mae'], metrics['amount']['mae']],
                   marker_color=['#e74c3c', '#f39c12']),
            row=2, col=2
        )
        
        fig.update_layout(
            title='📈 모델 성능 대시보드',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def save_visualizations(self, file_id: str, charts: dict, chart_data: dict):
        """
        시각화를 MongoDB에 저장
        """
        viz_doc = {
            'viz_id': f"viz_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'file_id': file_id,
            'user_id': self.mongo.files.find_one({'file_id': file_id})['user_id'],
            'charts': charts,
            'chart_data': chart_data,
            'created_at': datetime.now()
        }
        
        self.mongo.db['visualizations'].insert_one(viz_doc)
        print(f"💾 시각화 저장 완료: {viz_doc['viz_id']}")
    
    def get_visualizations(self, file_id: str):
        """
        저장된 시각화 조회
        """
        viz = self.mongo.db['visualizations'].find_one(
            {'file_id': file_id},
            sort=[('created_at', -1)]
        )
        return viz

