# app.py (Refactored)

import gradio as gr
import os
import sys
from dotenv import load_dotenv

# .env 파일로부터 환경변수를 로드합니다.
load_dotenv()

import nltk
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import VECTOR_STORE_PATH, OLLAMA_BASE_URL
from data_loader import load_documents, create_parent_document_retriever
from ui_layout import create_ui

# --- NLTK 데이터 자동 설치 로직 ---
def setup_nltk():
    """
    앱 실행에 필요한 NLTK 데이터 패키지가 있는지 확인하고,
    없으면 자동으로 다운로드합니다.
    """
    required_packages = {
        "tokenizers/punkt": "punkt",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    }
    
    print("NLTK 데이터 패키지를 확인합니다...")
    for path, package_id in required_packages.items():
        try:
            nltk.data.find(path)
            print(f" -> NLTK 패키지 '{package_id}'는 이미 설치되어 있습니다.")
        except LookupError:
            print(f" -> NLTK 패키지 '{package_id}'를 찾을 수 없습니다. 다운로드를 시작합니다...")
            nltk.download(package_id)
            print(f" -> NLTK 패키지 '{package_id}' 다운로드 완료.")
    print("NLTK 데이터 준비 완료.")

# --- [수정] 실행 환경에 맞춰 Embedding 장치를 자동으로 선택하는 로직 ---
def get_optimal_device():
    """실행 환경에 맞는 최적의 Pytorch 장치(device)를 감지하여 반환합니다."""
    if torch.cuda.is_available():
        print(" -> 최적 장치 감지: CUDA (NVIDIA GPU)")
        return 'cuda'
    if torch.backends.mps.is_available():
        print(" -> 최적 장치 감지: MPS (Apple Silicon GPU)")
        return 'mps'
    print(" -> 최적 장치 감지: CPU")
    return 'cpu'

def initialize_retriever(embeddings):
    print("Retriever 초기화를 시작합니다...")
    documents = load_documents()
    
    # Create an empty FAISS vector store
    vectorstore = FAISS.from_texts(texts=[""], embedding=embeddings)
    
    retriever = create_parent_document_retriever(vectorstore, documents)
    print("Retriever 준비 완료.")
    return retriever


def check_ollama_connection():
    """앱 시작 시 Ollama 서버와의 연결을 확인합니다."""
    try:
        import ollama
        # timeout을 짧게 설정하여 응답이 없을 때 오래 기다리지 않도록 함
        ollama.list(timeout=2) 
        print("✅ Ollama 서버와 성공적으로 연결되었습니다.")
        return True
    except Exception as e:
        print("\n" + "="*50)
        print("❌ 경고: Ollama 서버에 연결할 수 없습니다.")
        print("   Ollama가 설치되어 있고 실행 중인지 확인해주세요.")
        print(f"   오류 상세: {e}")
        print("="*50 + "\n")
        # Gradio 앱 자체는 실행되도록 하되, 경고를 명확히 표시
        gr.Warning("Ollama 서버에 연결할 수 없습니다. 로컬 모델 기능이 작동하지 않을 수 있습니다.")
        return False

if __name__ == "__main__":
    # --- 1. 전역 객체 초기화 ---
    setup_nltk()

    if "GOOGLE_API_KEY" not in os.environ or os.environ["GOOGLE_API_KEY"] == "YOUR_API_KEY_HERE":
        print("오류: GOOGLE_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    if OLLAMA_BASE_URL:
        ollama_host = OLLAMA_BASE_URL.replace("http://", "").replace("https://", "")
        os.environ['OLLAMA_HOST'] = ollama_host
        print(f"Ollama 서버 주소를 '{os.environ['OLLAMA_HOST']}'로 설정합니다.")

    DEVICE = get_optimal_device()
    print(f"전역 임베딩 모델로 HuggingFace 모델(Granite-Embedding-278m)을 생성합니다. (사용 장치: {DEVICE.upper()})")
    EMBEDDINGS = HuggingFaceEmbeddings(
        model_name="ibm-granite/granite-embedding-278m-multilingual",
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )

    RETRIEVER = initialize_retriever(EMBEDDINGS)

    # --- 2. UI 생성 및 실행 ---
    check_ollama_connection() # Gradio 앱 실행 전 연결 확인
    
    retriever_state = gr.State(RETRIEVER)
    
    demo = create_ui(retriever_state, EMBEDDINGS)
    
    print("Gradio 앱을 시작합니다. 웹 브라우저에서 다음 주소로 접속하세요.")
    demo.launch()