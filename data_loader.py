# data_loader.py

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Document 클래스를 임포트합니다.
import os
import re
import glob
from config import BASE_DIR, DATA_DIRS # [수정] config에서 경로 설정 가져오기

def parse_metadata_from_path(file_path):
    """파일 경로를 분석하여 구조화된 메타데이터를 추출합니다."""
    # [수정] 기본 메타데이터 구조를 미리 정의
    metadata = {
        "source": file_path,
        "status": "분류 안됨",
        "doc_type": "미분류",
        "company": "알 수 없음",
        "job_title": "알 수 없음",
        "date": None
    }
    
    try:
        parts = file_path.split(os.sep)
        filename = os.path.basename(file_path)

        # 1. 상태 정보 추출 (폴더 이름 기반)
        if 'Success' in parts or '성공' in parts:
            metadata['status'] = '최종 합격'
        elif '04_Passed_1st_Interview' in parts:
            metadata['status'] = '1차 면접 합격'
        elif '03_Passed_Screening' in parts:
            metadata['status'] = '서류 합격'
        elif 'Fail' in parts or '실패' in parts:
            metadata['status'] = '불합격'
        elif '02_Submitted' in parts:
            metadata['status'] = '제출 완료'
        elif '01_Working_Drafts' in parts:
            metadata['status'] = '작성 중'

        # 2. 문서 종류 및 상세 정보 추출 (파일명 기반)
        match = re.match(r'^\[([^\]]*)\]\s*([^_]*)_(.*)(?:_(\d{8}))?\.md$', filename, re.IGNORECASE)
        if match:
            metadata['doc_type'] = match.group(1).strip()
            metadata['company'] = match.group(2).strip()
            metadata['job_title'] = match.group(3).strip()
            if match.group(4):
                metadata['date'] = match.group(4).strip()
        else:
            if '기업분석' in filename: metadata['doc_type'] = '기업 분석'
            elif '산업동향' in filename: metadata['doc_type'] = '산업 동향'
            elif '기술트렌드' in filename: metadata['doc_type'] = '기술 트렌드'
            
    except Exception as e:
        print(f"[경고] 메타데이터 파싱 중 오류 발생 (파일: {file_path}): {e}. 기본 메타데이터를 사용합니다.")
        # [수정] 오류가 발생해도 이미 설정된 기본값이 유지되므로 pass 대신 명시적으로 경고 출력
    
    return metadata

def load_documents():
    """지정된 데이터 디렉토리에서 문서를 로드하고 제외 패턴을 적용합니다."""
    all_files = []
    for data_dir in DATA_DIRS:
        dir_path = os.path.join(BASE_DIR, data_dir)
        if os.path.isdir(dir_path):
            all_files.extend(glob.glob(os.path.join(dir_path, "**/*.md"), recursive=True))
        else:
            print(f"[경고] 설정된 데이터 디렉토리를 찾을 수 없습니다: {dir_path}")

    exclude_patterns = [
        "**/faiss_index/**", 
        "**/venv*/**", # [수정] venv, venv-gpu 등 다양한 이름의 가상환경 폴더 제외
        "**/__pycache__/**", 
        "**/5_Web_Search_Cache/**",
        "**/1_Resumes_and_CVs/01_Working_Drafts/**"
    ]
    
    excluded_files = set()
    for pattern in exclude_patterns:
        # [수정] BASE_DIR 기준으로 제외 패턴 검색
        excluded_files.update(glob.glob(os.path.join(BASE_DIR, pattern), recursive=True))
        
    files_to_load = [f for f in all_files if f not in excluded_files]
    
    documents = []
    print(f"총 {len(files_to_load)}개의 파일을 로드합니다...")
    for i, file_path in enumerate(files_to_load):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            doc = Document(page_content=content, metadata={"source": file_path})
            documents.append(doc)

            if (i + 1) % 10 == 0:
                print(f"  ... {i + 1}/{len(files_to_load)}개 처리 완료")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    print(f"총 {len(documents)}개의 문서를 성공적으로 불러왔습니다.")

    for doc in documents:
        new_metadata = parse_metadata_from_path(doc.metadata['source'])
        doc.metadata.update(new_metadata)
        
    print("모든 문서의 메타데이터 파싱을 완료했습니다.")
    return documents

def split_documents(documents):
    """불러온 문서를 청크 단위로 분할합니다."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # 청크 크기
        chunk_overlap=200   # 청크 간 겹치는 글자 수
    )
    splitted_docs = text_splitter.split_documents(documents)
    print(f"문서를 총 {len(splitted_docs)}개의 청크로 분할했습니다.")
    return splitted_docs