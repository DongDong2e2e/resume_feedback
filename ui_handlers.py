# ui_handlers.py

import gradio as gr
import os
import sys
import json
import re
import glob
from datetime import datetime

# 기존 app.py에서 사용하던 모듈들을 모두 import 합니다.
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_handler import llm_gemini_flash_normal
from prompts import PDF_CONVERTER_PROMPT_TEMPLATE, TOPIC_SUGGESTION_PROMPT_TEMPLATE
from orchestrator import generate_report_and_feedback, get_follow_up
from config import BASE_DIR, DATA_DIRS, AI_GENERATED_NOTES_PATH
from data_loader import load_documents, split_documents
from vector_store_manager import create_and_save_vector_store

# --- 유틸리티 함수 ---
def get_md_files():
    """[수정] DATA_DIRS에 지정된 폴더 내의 모든 .md 파일을 찾아서 반환합니다."""
    md_files = []
    # [수정] data_loader.py와 동일한 제외 로직을 일부 적용 (UI 표시용) 
    exclude_patterns = ["**/01_Working_Drafts/**"]
    
    for data_dir in DATA_DIRS:
        dir_path = os.path.join(BASE_DIR, data_dir)
        if os.path.isdir(dir_path):
            files_in_dir = glob.glob(os.path.join(dir_path, "**/*.md"), recursive=True)
            
            # 제외 패턴 적용
            excluded_files = set()
            for pattern in exclude_patterns:
                excluded_files.update(glob.glob(os.path.join(dir_path, pattern), recursive=True))
            
            valid_files = [f for f in files_in_dir if f not in excluded_files]

            # 경로를 BASE_DIR 기준으로 상대 경로로 변경
            for f in valid_files:
                md_files.append(os.path.relpath(f, BASE_DIR))
                
    return sorted(md_files)

def read_file_content(file_path_relative):
    """[수정] BASE_DIR 기준의 상대 경로를 받아 파일 내용을 읽어 반환합니다."""
    if not file_path_relative:
        return ""
    base_path = os.path.abspath(BASE_DIR)
    file_path = os.path.abspath(os.path.join(base_path, file_path_relative))
    if not file_path.startswith(base_path):
        raise gr.Error("허용되지 않은 파일 경로입니다.")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise gr.Error("파일을 찾을 수 없습니다.")
    except Exception as e:
        raise gr.Error(f"파일을 읽는 중 오류 발생: {e}")

def save_file_content(file_path_relative, content):
    """[수정] BASE_DIR 기준의 상대 경로를 받아 파일을 저장합니다."""
    if not file_path_relative:
        raise gr.Error("파일이 선택되지 않았습니다.")
    
    base_path = os.path.abspath(BASE_DIR)
    file_path = os.path.abspath(os.path.join(base_path, file_path_relative))
    
    if not file_path.startswith(base_path):
        raise gr.Error("허용되지 않은 파일 경로입니다.")
        
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"'{file_path_relative}' 파일이 성공적으로 저장되었습니다."
    except Exception as e:
        raise gr.Error(f"파일 저장 중 오류 발생: {e}")

# --- 벡터 저장소 핵심 로직 ---
def _rebuild_store_core(vector_store_path, embeddings):
    print("벡터 DB 재생성을 시작합니다...")
    all_documents = load_documents()
    if not all_documents:
        print("경고: 문서를 찾을 수 없어 벡터 DB를 생성할 수 없습니다.")
        return None 
    doc_chunks = split_documents(all_documents)
    vector_store = create_and_save_vector_store(doc_chunks, vector_store_path, embeddings)
    print("벡터 DB 재생성 완료.")
    return vector_store

def rebuild_vector_db_for_ui(vector_store_path, embeddings, VECTOR_STORE):
    """(복원) UI용 벡터 DB 새로고침 함수, 전역 변수 사용"""
    try:
        VECTOR_STORE = _rebuild_store_core(vector_store_path, embeddings)
        if VECTOR_STORE is None:
            return "벡터 DB 생성 중 오류가 발생했습니다. 터미널 로그를 확인해주세요."
        return "벡터 데이터베이스가 성공적으로 업데이트되었습니다."
    except Exception as e:
        print(f"벡터 DB 재생성 중 오류 발생: {e}")
        raise gr.Error(f"벡터 DB 업데이트 실패: {e}")

# --- AI 기반 자료 자동화 핸들러 ---
def suggest_topic_from_file(files, progress=gr.Progress()):
    """파일 객체 목록을 받아 파일명 목록을 표시하고, 첫 번째 파일로 주제를 추천합니다."""
    if not files:
        return "", ""

    filenames = ", ".join([os.path.basename(f.name) for f in files])
    first_file = files[0]
    progress(0.3, desc=f"'{os.path.basename(first_file.name)}' 파일을 분석 중입니다...")
    try:
        loader = UnstructuredFileLoader(first_file.name)
        raw_text = loader.load()[0].page_content

        progress(0.7, desc="AI가 핵심 주제를 분석 중입니다...")
        suggestion_prompt = ChatPromptTemplate.from_template(TOPIC_SUGGESTION_PROMPT_TEMPLATE)
        suggestion_chain = suggestion_prompt | llm_gemini_flash_normal | StrOutputParser()
        suggested_topic = suggestion_chain.invoke({"raw_text": raw_text})

        return filenames, suggested_topic.strip()
    except Exception as e:
        gr.Warning(f"주제 추천 중 오류 발생: {e}")
        return filenames, ""

def handle_file_upload(files, file_context, vector_store_path, embeddings, progress=gr.Progress(track_tqdm=True)):
    if not files:
        raise gr.Error("분석할 파일을 먼저 업로드해주세요.")

    processed_files = []
    total_files = len(files)
    
    for i, file_obj in enumerate(files):
        try:
            progress(i / total_files, desc=f"({i+1}/{total_files}) '{os.path.basename(file_obj.name)}' 처리 중...")
            
            loader = UnstructuredFileLoader(file_obj.name)
            raw_text = loader.load()[0].page_content
            
            current_context = file_context 
            
            if not file_context: 
                print(f" -> '{os.path.basename(file_obj.name)}' 파일의 개별 주제를 생성합니다.")
                suggestion_prompt = ChatPromptTemplate.from_template(TOPIC_SUGGESTION_PROMPT_TEMPLATE)
                suggestion_chain = suggestion_prompt | llm_gemini_flash_normal | StrOutputParser()
                individual_topic = suggestion_chain.invoke({"raw_text": raw_text})
                current_context = individual_topic.strip()

            converter_prompt = ChatPromptTemplate.from_template(PDF_CONVERTER_PROMPT_TEMPLATE)
            converter_chain = converter_prompt | llm_gemini_flash_normal | StrOutputParser()
            llm_output = converter_chain.invoke({"file_context": current_context, "raw_text": raw_text})

            lines = llm_output.strip().split('\n')
            new_filename = ""
            content_lines = []
            filename_found = False

            for j, line in enumerate(lines):
                stripped_line = line.strip().strip("`")
                if stripped_line.endswith(".md") and not filename_found:
                    new_filename = stripped_line
                    content_lines = lines[j+1:]
                    filename_found = True
                    break
            
            if not filename_found:
                raise ValueError(f"LLM 응답에서 .md 파일명을 찾지 못했습니다.")
            
            new_content = '\n'.join(content_lines).strip()
            
            relative_path = os.path.join(os.path.basename(AI_GENERATED_NOTES_PATH), new_filename)
            save_file_content(relative_path, new_content)
            processed_files.append(new_filename)

        except Exception as e:
            gr.Warning(f"'{os.path.basename(file_obj.name)}' 처리 중 오류 발생: {e}")
            continue

    if not processed_files:
        raise gr.Error("모든 파일 처리 중 오류가 발생했습니다.")

    progress(1.0, desc="벡터 DB에 반영 중입니다...")
    rebuild_status = rebuild_vector_db_for_ui(vector_store_path, embeddings, None) # VECTOR_STORE is not available here
    final_message = f"총 {len(processed_files)}개 파일의 노트 생성을 완료했습니다.\n{rebuild_status}"
    return final_message, gr.Dropdown(choices=get_md_files(), value=os.path.join(os.path.basename(AI_GENERATED_NOTES_PATH), processed_files[-1]))

def manual_save_button_handler(file_path_relative, content):
    status_message = save_file_content(file_path_relative, content)
    return status_message, gr.Dropdown(choices=get_md_files(), value=file_path_relative)

def save_results_to_file(report, eval_md, itemized_md, rewrite_md, company, job_title):
    """생성된 리포트와 피드백을 하나의 마크다운 파일로 저장합니다."""
    if not any([report, eval_md, itemized_md, rewrite_md]):
        gr.Info("저장할 내용이 없습니다.")
        return

    from datetime import datetime
    
    save_dir = "6_Consulting_Results"
    os.makedirs(save_dir, exist_ok=True)
    
    today_str = datetime.now().strftime('%Y%m%d')
    safe_company = "".join(c for c in company if c.isalnum())
    safe_job = "".join(c for c in job_title if c.isalnum())
    filename = f"[{today_str}]_{safe_company}_{safe_job}_컨설팅결과.md"
    filepath = os.path.join(save_dir, filename)

    full_content = f"# {company} - {job_title} AI 컨설팅 결과\n\n"
    full_content += f"## 사전 브리핑 리포트\n\n---
{report}\n\n"
    full_content += f"## 종합 분석\n\n---
{eval_md}\n\n"
    full_content += f"## 항목별 상세 피드백\n\n---
{itemized_md}\n\n"
    full_content += f"## AI 추천 수정본\n\n---
{rewrite_md}\n\n"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        gr.Info(f"'{filepath}'에 성공적으로 저장했습니다.")
    except Exception as e:
        gr.Warning(f"파일 저장 중 오류 발생: {e}")

def report_and_feedback_interface(company_name, job_title, job_description, my_draft, vector_store):
    if not all([company_name, job_title, job_description, my_draft]):
        raise gr.Error("모든 필드를 입력해주세요.")
    if not vector_store:
        raise gr.Error("벡터 저장소가 비어있습니다. 자료를 추가하고 DB를 다시 만들어주세요.")

    yield "...", "...", "...", "...", gr.update(visible=False), [], "", gr.Button(value="생성 중...", interactive=False), gr.update(visible=False)
    
    initial_context = f"회사명: {company_name}\n직무명: {job_title}"
    response_generator = generate_report_and_feedback(vector_store, job_description, my_draft, company_name, job_title)

    final_report, final_feedback_json_str = "", ""
    for response_data in response_generator:
        final_report = response_data["report"]
        final_feedback_json_str = response_data["feedback"]
        yield final_report, final_feedback_json_str, "...", "...", gr.update(visible=False), [], initial_context, gr.Button(value="생성 중...", interactive=False), gr.update(visible=False)

    overall_eval_md = "### 📊 종합 분석\n\n분석 내용을 생성하지 못했습니다."
    itemized_md = "### ✍️ 항목별 상세 피드백\n\n피드백을 생성하지 못했습니다."
    rewrite_md = "### ✨ AI 추천 수정본\n\n수정본을 생성하지 못했습니다."

    try:
        match = re.search(r'```json\s*([\s\S]+?)\s*```', final_feedback_json_str)
        json_str = match.group(1) if match else final_feedback_json_str
        
        print("--- 최종 LLM 응답 (JSON 파싱 전) ---")
        print(json_str)
        print("-------------------------------------")

        data = json.loads(json_str)

        if overall_eval := data.get('overall_evaluation'):
            overall_eval_md = overall_eval.replace('\\n', '\n')
        else:
            overall_eval_md = "### 📊 종합 분석\n\nAI가 종합 분석을 생성하지 않았습니다."

        if item_list := data.get('itemized_feedback'):
            table = "| 원본 내용 | 💡 AI 컨설턴트 피드백 |\n|---|---|"
            for item in item_list:
                original = item.get('original_paragraph', '').replace('\n', '<br>')
                feedback = item.get('feedback', '').replace('\n', '<br>')
                table += f"| {original} | {feedback} |\n"
            itemized_md = "### ✍️ 항목별 상세 피드백\n\n" + table
        else:
            itemized_md = "### ✍️ 항목별 상세 피드백\n\nAI가 항목별 피드백을 생성하지 않았습니다."

        if full_rewrite := data.get('full_rewrite'):
            rewrite_md = f"### ✨ AI 추천 수정본\n\n---\n\n{full_rewrite}"
        else:
            rewrite_md = "### ✨ AI 추천 수정본\n\nAI가 전체 수정본을 생성하지 않았습니다."

    except Exception as e:
        print(f"[경고] 최종 응답 파싱 오류: {e}")
        error_message = f"### ⚠️ 오류\n\nLLM 응답 처리 중 오류가 발생했습니다: {e}\n\n**원본:**\n```\n{final_feedback_json_str}\n```"
        overall_eval_md = error_message
        itemized_md = error_message
        rewrite_md = error_message

    initial_chat_history = [{'role': 'assistant', 'content': f"{company_name} {job_title} 직무 리포트입니다. 궁금한 점을 질문해주세요!"}]
    yield final_report, overall_eval_md, itemized_md, rewrite_md, gr.update(visible=True), initial_chat_history, initial_context, gr.Button(value="리포트 및 피드백 받기", interactive=True), gr.update(visible=True)

def handle_chat_submission(question, history, initial_context, vector_store):
    if vector_store is None:
        raise gr.Error("오류: 벡터 저장소가 초기화되지 않았습니다.")
    
    history.append({"role": "user", "content": question})
    response_stream = get_follow_up(question, history, vector_store, initial_context)
    
    bot_message = ""
    history.append({"role": "assistant", "content": ""})
    for chunk in response_stream:
        bot_message += chunk
        history[-1]["content"] = bot_message
        yield history, ""
