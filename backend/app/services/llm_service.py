from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # 初始化模型
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.3, 
                convert_system_message_to_human=True
            )
            logger.info("LLM Service initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise e

    def generate_response(self, prompt: str) -> str:
        """
        發送 Prompt 給 LLM 並取得純文字回應
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I am having trouble thinking right now."

llm_service = LLMService()