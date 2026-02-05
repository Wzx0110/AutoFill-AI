import requests
import os

# 後端網址 
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api")

class APIClient:
    @staticmethod
    def upload_reference(file_obj):
        """
        呼叫後端上傳 PDF
        """
        try:
            files = {"file": (file_obj.name, file_obj, "application/pdf")}
            response = requests.post(f"{BACKEND_URL}/upload-reference", files=files)
            response.raise_for_status() 
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    @staticmethod
    def query_knowledge(question: str):
        """
        呼叫後端進行 RAG 問答
        """
        try:
            payload = {"question": question}
            response = requests.post(f"{BACKEND_URL}/query", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

api_client = APIClient()