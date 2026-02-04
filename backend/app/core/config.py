from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 專案基本設定
    PROJECT_NAME: str = "AutoFill AI"
    VERSION: str = "0.1.0"
    
    # 敏感資料 (自動從 .env 讀取)
    GOOGLE_API_KEY: str
    LLAMA_CLOUD_API_KEY: str 
    QDRANT_URL: str = "http://localhost:6333"
    
    # 定義模型名稱
    LLM_MODEL: str = "gemini-3-flash-preview" 
    # Embedding 模型 
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    class Config:
        env_file = ".env"
        extra = "ignore" # 忽略 .env 中多餘變數

settings = Settings()