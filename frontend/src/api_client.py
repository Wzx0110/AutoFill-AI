import requests
import os
import json

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
        
    @staticmethod
    def extract_data(fields: list):
        """
        呼叫後端 Auto-Fill API
        fields: List[Dict], e.g., [{"key": "name", "description": "...", "data_type": "string"}]
        """
        try:
            payload = {
                "collection_name": "reference_docs",
                "fields": fields
            }
            
            response = requests.post(f"{BACKEND_URL}/extract", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
        
    @staticmethod
    def analyze_form(file_obj):
        try:
            files = {"file": (file_obj.name, file_obj, "application/pdf")} # 或 word
            response = requests.post(f"{BACKEND_URL}/analyze-form", files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
        
    @staticmethod
    def generate_filled_file(file_obj, results_list):
        try:
            # 準備 multipart/form-data
            files = {"file": (file_obj.name, file_obj, file_obj.type)}
            data = {"results_json": json.dumps(results_list)}
            
            response = requests.post(f"{BACKEND_URL}/generate-file", files=files, data=data)
            
            if response.status_code == 200:
                return response.content # 回傳二進制檔案內容
            else:
                return {"error": f"Failed: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

api_client = APIClient()