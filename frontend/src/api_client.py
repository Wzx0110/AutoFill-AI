import requests
import os

# 後端網址 
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api")

class APIClient:
    @staticmethod
    def upload_reference(file_objs):
        """
        支援多檔上傳
        file_objs: List of files
        """
        try:
            files_payload = [
                ("files", (file.name, file, "application/pdf")) for file in file_objs
            ]
            response = requests.post(f"{BACKEND_URL}/upload-reference", files=files_payload)
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