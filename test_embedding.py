# test_embedding.py
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
# [수정] config 대신 직접 URL을 사용합니다.
# from config import EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL 

# .env 파일은 이 테스트에 직접적인 영향을 주지 않지만, 원래 코드와 환경을 맞추기 위해 포함합니다.
load_dotenv() 

print("--- 임베딩 모델 단독 테스트 시작 ---")
# [수정] 모델과 URL을 직접 하드코딩합니다.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL_NAME = "nomic-embed-text"

print(f"설정된 Ollama 서버 주소: {OLLAMA_BASE_URL}")
print(f"사용할 임베딩 모델: {EMBEDDING_MODEL_NAME}")

try:
    # vector_store_manager.py와 동일한 방식으로 임베딩 모델을 초기화합니다.
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME, 
        base_url=OLLAMA_BASE_URL
    )
    
    test_text = "이것은 임베딩 기능을 테스트하기 위한 간단한 문장입니다."
    print(f"\n테스트 문장: {test_text}")
    
    # 단일 문장에 대한 임베딩을 시도합니다.
    vector = embeddings.embed_query(test_text)
    
    print("\n[성공] 임베딩 벡터 생성 완료!")
    print(f"생성된 벡터의 일부: {vector[:5]}...")
    print(f"벡터 차원: {len(vector)}")

except Exception as e:
    print(f"\n[오류] 임베딩 테스트 중 오류 발생:")
    print(e)

finally:
    print("\n--- 테스트 종료 ---")
