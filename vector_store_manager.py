# vector_store_manager.py

# OllamaEmbeddings는 이제 app.py에서 직접 임포트하므로 여기서 삭제합니다.
from langchain_community.vectorstores import FAISS
import os
import time
from tqdm import tqdm

def create_and_save_vector_store(chunks, path, embeddings, batch_size=32):
    """
    텍스트 청크로부터 벡터 저장소를 생성하고 저장합니다.
    이제 embeddings 객체를 파라미터로 직접 받습니다.
    """
    print(f"벡터 저장소 생성 시작... (총 청크 수: {len(chunks)}, 배치 크기: {batch_size})")

    if not chunks:
        print("경고: 문서 청크가 없어 벡터 저장소를 생성할 수 없습니다.")
        return None

    try:
        print(" -> 1단계: 첫 번째 배치로 벡터 저장소 초기화 중...")
        first_batch = chunks[:batch_size]
        vectorstore = FAISS.from_documents(documents=first_batch, embedding=embeddings)
        print(" -> 초기화 완료.")

        remaining_chunks = chunks[batch_size:]
        for i in tqdm(range(0, len(remaining_chunks), batch_size), desc=" -> 2단계: 나머지 배치 처리 중"):
            batch = remaining_chunks[i:i + batch_size]
            if batch:
                vectorstore.add_documents(documents=batch)
                time.sleep(0.1)

        print("\n -> 3단계: 최종 벡터 저장소 저장 중...")
        original_cwd = os.getcwd()
        parent_dir = os.path.dirname(path)
        index_name = os.path.basename(path)
        
        try:
            os.chdir(parent_dir)
            vectorstore.save_local(index_name)
            print(f"벡터 저장소를 '{path}'에 성공적으로 생성하고 저장했습니다.")
        finally:
            os.chdir(original_cwd)
            
        return vectorstore

    except Exception as e:
        print(f"\n[오류] 배치 처리 중 심각한 오류 발생: {e}")
        return None

def load_vector_store(path, embeddings):
    """
    저장된 벡터 저장소를 불러옵니다.
    이제 embeddings 객체를 파라미터로 직접 받습니다.
    """
    if not os.path.exists(os.path.join(path, "index.faiss")):
        return None
    
    original_cwd = os.getcwd()
    parent_dir = os.path.dirname(path)
    index_name = os.path.basename(path)
    
    vectorstore = None
    try:
        os.chdir(parent_dir)
        vectorstore = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
        print(f"'{path}'에서 벡터 저장소를 불러왔습니다.")
    finally:
        os.chdir(original_cwd)
        
    return vectorstore