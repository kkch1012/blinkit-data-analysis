"""
LLM 챗봇 서비스 - 예측 결과를 해석하고 질문에 답변
"""

from mongodb_schema import MongoDBService
from datetime import datetime
import json
from typing import List, Dict, Optional

# LLM API 선택 (OpenAI, Anthropic, Google 등)
try:
    from openai import OpenAI
    LLM_PROVIDER = "openai"
except:
    try:
        import anthropic
        LLM_PROVIDER = "anthropic"
    except:
        LLM_PROVIDER = None


class LLMChatbotService:
    """LLM 챗봇 서비스"""
    
    def __init__(self, api_key: str = None, provider: str = "openai"):
        """
        Args:
            api_key: LLM API 키
            provider: "openai", "anthropic", "google" 등
        """
        self.mongo = MongoDBService()
        self.provider = provider
        
        if provider == "openai" and api_key:
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic" and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            print("⚠️ LLM API가 설정되지 않았습니다. 모의 응답을 사용합니다.")
            self.client = None
    
    def generate_insights(self, predictions: dict, metrics: dict, 
                         feature_importance: dict = None) -> str:
        """
        예측 결과를 LLM이 해석하여 인사이트 생성
        """
        # 프롬프트 구성
        prompt = f"""
당신은 데이터 분석 전문가입니다. 다음 수요 예측 분석 결과를 바탕으로 인사이트를 생성해주세요.

[수량 예측 결과]
- 평균 예측 수량: {predictions['quantity']['avg']:.2f}개
- 예측 정확도: {metrics['quantity']['accuracy']:.2f}%
- MAE (평균 절대 오차): {metrics['quantity']['mae']:.4f}
- R² Score: {metrics['quantity']['r2']:.4f}

[금액 예측 결과]
- 평균 예측 금액: {predictions['amount']['avg']:,.0f}원
- 예측 정확도: {metrics['amount']['accuracy']:.2f}%
- MAE (평균 절대 오차): {metrics['amount']['mae']:.4f}
- R² Score: {metrics['amount']['r2']:.4f}

[예측 일정]
{self._format_predictions(predictions)}

위 결과를 바탕으로 다음을 포함하여 한국어로 친절하게 설명해주세요:

1. **주요 인사이트 3가지** (가장 중요한 발견사항)
2. **비즈니스 관점 해석** (이 예측이 비즈니스에 어떤 의미인지)
3. **행동 권장사항** (재고 관리, 마케팅 등)
4. **주의사항** (예측의 한계나 불확실성)

친절하고 전문적인 톤으로 작성해주세요.
"""
        
        if self.client:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "당신은 데이터 분석 전문가입니다. 수요 예측 결과를 해석하고 비즈니스 인사이트를 제공합니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                return response.choices[0].message.content
            elif self.provider == "anthropic":
                message = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text
        else:
            # 모의 응답 (API 없을 때)
            return self._mock_insights(predictions, metrics)
    
    def answer_question(self, question: str, file_id: str, 
                       predictions: dict = None, 
                       visualizations: dict = None,
                       chat_history: List[Dict] = None) -> str:
        """
        사용자 질문에 답변
        """
        # 컨텍스트 구성
        context_parts = []
        
        if predictions:
            context_parts.append(f"""
[예측 결과]
수량: 평균 {predictions['quantity']['avg']:.2f}개
금액: 평균 {predictions['amount']['avg']:,.0f}원
""")
        
        if chat_history:
            context_parts.append(f"""
[이전 대화]
{self._format_chat_history(chat_history[-5:])}  # 최근 5개만
""")
        
        prompt = f"""
당신은 수요 예측 분석 챗봇입니다. 사용자의 질문에 답변해주세요.

{''.join(context_parts)}

[사용자 질문]
{question}

위 정보를 바탕으로 사용자의 질문에 정확하고 친절하게 답변해주세요.
- 구체적인 숫자와 데이터를 언급하세요
- 불확실한 부분은 솔직하게 말하세요
- 한국어로 답변하세요
"""
        
        if self.client:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "당신은 수요 예측 분석 챗봇입니다. 사용자의 질문에 정확하고 친절하게 답변합니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                answer = response.choices[0].message.content
            elif self.provider == "anthropic":
                message = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = message.content[0].text
        else:
            answer = self._mock_answer(question)
        
        # 대화 저장
        self.save_message(file_id, "user", question)
        self.save_message(file_id, "assistant", answer)
        
        return answer
    
    def _format_predictions(self, predictions: dict) -> str:
        """예측 결과를 포맷팅"""
        text = "\n[수량 예측]\n"
        for date, qty in zip(predictions['quantity']['dates'], 
                            predictions['quantity']['predicted']):
            text += f"  {date}: {qty:.2f}개\n"
        
        text += "\n[금액 예측]\n"
        for date, amt in zip(predictions['amount']['dates'], 
                            predictions['amount']['predicted']):
            text += f"  {date}: {amt:,.0f}원\n"
        
        return text
    
    def _format_chat_history(self, history: List[Dict]) -> str:
        """대화 이력을 포맷팅"""
        text = ""
        for msg in history:
            role = "사용자" if msg['role'] == 'user' else "챗봇"
            text += f"{role}: {msg['content']}\n"
        return text
    
    def save_message(self, file_id: str, role: str, content: str):
        """
        대화 메시지 저장
        """
        # 기존 채팅 찾기 또는 생성
        chat = self.mongo.db['chat_history'].find_one(
            {'file_id': file_id},
            sort=[('created_at', -1)]
        )
        
        if not chat:
            chat = {
                'chat_id': f"chat_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'file_id': file_id,
                'user_id': self.mongo.files.find_one({'file_id': file_id})['user_id'],
                'messages': [],
                'created_at': datetime.now()
            }
        
        # 메시지 추가
        chat['messages'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })
        
        # 저장 또는 업데이트
        if '_id' in chat:
            self.mongo.db['chat_history'].update_one(
                {'_id': chat['_id']},
                {'$set': {'messages': chat['messages']}}
            )
        else:
            self.mongo.db['chat_history'].insert_one(chat)
    
    def get_chat_history(self, file_id: str, limit: int = 50) -> List[Dict]:
        """
        대화 이력 조회
        """
        chat = self.mongo.db['chat_history'].find_one(
            {'file_id': file_id},
            sort=[('created_at', -1)]
        )
        
        if chat:
            return chat['messages'][-limit:]  # 최근 N개만
        return []
    
    def _mock_insights(self, predictions: dict, metrics: dict) -> str:
        """모의 인사이트 (API 없을 때)"""
        return f"""
안녕하세요! 수요 예측 분석 결과를 요약해드리겠습니다.

📊 **주요 인사이트**

1. **수량 예측**: 향후 7일 평균 {predictions['quantity']['avg']:.2f}개로 예상됩니다.
   - 예측 정확도: {metrics['quantity']['accuracy']:.2f}%
   - 이는 전반적으로 안정적인 판매 패턴을 보여줍니다.

2. **금액 예측**: 평균 {predictions['amount']['avg']:,.0f}원으로 예상됩니다.
   - 예측 정확도: {metrics['amount']['accuracy']:.2f}%
   - 수량 대비 금액 증가율을 모니터링하세요.

3. **권장사항**: 
   - 재고는 평균 수량의 1.2배 정도 준비하시는 것을 권장합니다.
   - 주말 판매량이 증가하는 패턴이 보이므로 주말 재고를 늘려보세요.

💡 **주의사항**: 예측 모델의 정확도는 {metrics['quantity']['accuracy']:.2f}%이므로, 
실제 수요와 차이가 있을 수 있습니다. 정기적으로 모델을 재학습하는 것을 권장합니다.
"""
    
    def _mock_answer(self, question: str) -> str:
        """모의 답변 (API 없을 때)"""
        return f"""
질문: {question}

죄송합니다. 현재 LLM API가 설정되지 않아 정확한 답변을 드리기 어렵습니다.
API 키를 설정하시면 더 자세한 분석과 답변을 제공할 수 있습니다.

일반적으로 수요 예측은 과거 데이터 패턴을 기반으로 하므로, 
계절성, 트렌드, 외부 요인(날씨, 이벤트 등)을 고려하는 것이 중요합니다.
"""

