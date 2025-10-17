# config.py
import os

# --- 경로 설정 ---
# 이 파일의 위치를 기준으로 절대 경로를 생성합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# RAG에 사용할 데이터가 있는 폴더 이름 목록
DATA_DIRS = [
    "1_Resumes_and_CVs",
    "2_Job_Postings_and_JD",
    "3_Company_and_Industry_Research",
    "4_Technical_Notes_and_Study",
]

# 생성된 벡터 DB를 저장할 폴더 이름
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "faiss_index")


# --- LLM 모델 설정 ---
# 답변 생성을 위한 모델 (Gemini Pro -> Gemini Flash -> Ollama 순서로 폴백)
GEMINI_PRO_MODEL_NAME = "gemini-2.5-pro"
GEMINI_FLASH_MODEL_NAME = "gemini-2.5-flash" # 새로운 Flash 모델
OLLAMA_MODEL_NAME = "hf.co/Liontix/Qwen3-8B-Gemini-2.5-Pro-Distill-GGUF:Q4_K_M" # 로컬에서 사용하는 Ollama 모델
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# --- 임베딩 모델 설정 ---
# 벡터화를 위해 사용할 모델 (Ollama)
# EMBEDDING_MODEL_NAME = "nomic-embed-text" # Ollama에서 사용하는 임베딩 모델

# --- AI 생성 노트 경로 ---
AI_GENERATED_NOTES_PATH = os.path.join(BASE_DIR, "_AI_Generated_Notes")
