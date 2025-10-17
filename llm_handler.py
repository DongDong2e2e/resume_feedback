# --- START OF FILE llm_handler.py ---
import os
import itertools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from config import (
    GEMINI_PRO_MODEL_NAME,
    GEMINI_FLASH_MODEL_NAME,
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL, # <-- 이미 import 되어 있습니다.
)

# --- LLM 인스턴스 생성 ---
# [수정] .env 파일에 저장된 GOOGLE_API_KEY를 직접 전달합니다.
llm_gemini_pro_normal = ChatGoogleGenerativeAI(
    model=GEMINI_PRO_MODEL_NAME, 
    temperature=0.7, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
llm_gemini_flash_normal = ChatGoogleGenerativeAI(
    model=GEMINI_FLASH_MODEL_NAME, 
    temperature=0.7, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
llm_ollama_normal = ChatOllama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.0)

llm_gemini_pro_stream = ChatGoogleGenerativeAI(
    model=GEMINI_PRO_MODEL_NAME, 
    temperature=0.7, 
    streaming=True, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
llm_gemini_flash_stream = ChatGoogleGenerativeAI(
    model=GEMINI_FLASH_MODEL_NAME, 
    temperature=0.7, 
    streaming=True, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
llm_ollama_stream = ChatOllama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.0, streaming=True)

# --- 폴백(Fallback) 로직 ---
def invoke_with_fallback(prompt, inputs, models, stream=False):
    """
    주어진 모델 목록을 순서대로 시도하여 프롬프트를 실행하고,
    성공하는 첫 번째 결과를 반환합니다.
    """
    output_parser = StrOutputParser() # 이 함수 안에서만 사용될 수 있으므로 지역적으로 선언 가능
    for model_instance in models:
        try:
            model_name = getattr(model_instance, 'model', getattr(model_instance, 'model_name', 'Unknown'))
            print(f"  - 모델 호출 시도: {model_name}")
            chain = prompt | model_instance | output_parser
            if stream:
                response_generator = chain.stream(inputs)
                first_chunk = next(response_generator)
                print(f"  - 스트리밍 호출 성공: {model_name}")
                return itertools.chain([first_chunk], response_generator), model_name
            else:
                response = chain.invoke(inputs)
                print(f"  - 일반 호출 성공: {model_name}")
                return response, model_name
        except Exception as e:
            print(f"  - 호출 실패: {model_name}. 오류: {e}")
            continue
    raise RuntimeError("모든 모델 호출에 실패했습니다.")
