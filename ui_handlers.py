# ui_handlers.py

import gradio as gr
import os
import sys
import json
import re
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
from data_loader import load_documents, create_parent_document_retriever
from langchain_community.vectorstores import FAISS

# --- 유틸리티 함수 ---
def get_md_files():
    """[수정] DATA_DIRS에 지정된 폴더 내의 모든 .md 파일을 찾아서 반환합니다."""
    md_files = []
    exclude_patterns = ["**/01_Working_Drafts/**"]
    
    for data_dir in DATA_DIRS:
        dir_path = os.path.join(BASE_DIR, data_dir)
        if os.path.isdir(dir_path):
            files_in_dir = glob.glob(os.path.join(dir_path, "**/*.md"), recursive=True)
            
            excluded_files = set()
            for pattern in exclude_patterns:
                excluded_files.update(glob.glob(os.path.join(dir_path, pattern), recursive=True))
            
            valid_files = [f for f in files_in_dir if f not in excluded_files]

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

def read_source_content_from_selection(selected_source):
    """ "[N] path/to/file.md" 형식의 문자열에서 실제 경로만 추출하여 내용을 읽습니다. """
    if not selected_source:
        return ""
    
    match = re.search(r'\]\s*(.*)', selected_source)
    if match:
        file_path_relative = match.group(1)
        return read_file_content(file_path_relative)
    
    return read_file_content(selected_source)

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

# --- Retriever 핵심 로직 ---
def _rebuild_retriever_core(embeddings):
    print("Retriever 재생성을 시작합니다...")
    documents = load_documents()
    if not documents:
        print("경고: 문서를 찾을 수 없어 Retriever를 생성할 수 없습니다.")
        return None
    
    vectorstore = FAISS.from_texts(texts=[""], embedding=embeddings)
    
    retriever = create_parent_document_retriever(vectorstore, documents)
    print("Retriever 재생성 완료.")
    return retriever

def rebuild_retriever_for_ui(embeddings):
    """UI용 Retriever 새로고침 함수. 새로운 Retriever 객체를 반환합니다."""
    try:
        new_retriever = _rebuild_retriever_core(embeddings)
        if new_retriever is None:
            return "Retriever 생성 중 오류가 발생했습니다.", None
        return "Retriever가 성공적으로 업데이트되었습니다.", new_retriever
    except Exception as e:
        print(f"Retriever 재생성 중 오류 발생: {e}")
        raise gr.Error(f"Retriever 업데이트 실패: {e}")

# --- AI 기반 자료 자동화 핸들러 ---
def suggest_topic_from_file(files, progress=gr.Progress()):
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

def handle_file_upload(files, file_context, embeddings, progress=gr.Progress(track_tqdm=True)):
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

    progress(1.0, desc="Retriever에 반영 중입니다...")
    rebuild_status, new_retriever = rebuild_retriever_for_ui(embeddings)
    
    final_message = f"총 {len(processed_files)}개 파일의 노트 생성을 완료했습니다.\n{rebuild_status}"
    
    dropdown_value = os.path.join(os.path.basename(AI_GENERATED_NOTES_PATH), processed_files[-1]) if processed_files else None
    return final_message, gr.Dropdown(choices=get_md_files(), value=dropdown_value), new_retriever

def manual_save_button_handler(file_path_relative, content):
    status_message = save_file_content(file_path_relative, content)
    return status_message, gr.Dropdown(choices=get_md_files(), value=file_path_relative)

def save_results_to_file(report, eval_md, itemized_md, rewrite_md, company, job_title):
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
    full_content += f"## 사전 브리핑 리포트\n\n---\n{report}\n\n"
    full_content += f"## 종합 분석\n\n---\n{eval_md}\n\n"
    full_content += f"## 항목별 상세 피드백\n\n---\n{itemized_md}\n\n"
    full_content += f"## AI 추천 수정본\n\n---\n{rewrite_md}\n\n"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        gr.Info(f"'{filepath}'에 성공적으로 저장했습니다.")
    except Exception as e:
        gr.Warning(f"파일 저장 중 오류 발생: {e}")

def report_and_feedback_interface(company_name, job_title, job_description, my_draft, retriever):
    if not all([company_name, job_title, job_description, my_draft]):
        raise gr.Error("모든 필드를 입력해주세요.")
    if not retriever:
        raise gr.Error("Retriever가 비어있습니다. 자료를 추가하고 DB를 다시 만들어주세요.")

    yield "...", "...", "...", "...", gr.update(visible=False), [], "", gr.Button(value="생성 중...", interactive=False), gr.update(visible=False), gr.update(choices=[], value=None)
    
    initial_context = f"회사명: {company_name}\n직무명: {job_title}"
    response_generator = generate_report_and_feedback(retriever, job_description, my_draft, company_name, job_title)

    final_report, final_feedback_json_str = "", ""
    final_sources = []
    for response_data in response_generator:
        final_report = response_data["report"]
        final_feedback_json_str = response_data["feedback"]
        final_sources = response_data["sources"]

        yield final_report, final_feedback_json_str, "...", "...", gr.update(visible=False), [], initial_context, gr.Button(value="생성 중...", interactive=False), gr.update(visible=False), gr.update(choices=final_sources, value=None)

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
    yield final_report, overall_eval_md, itemized_md, rewrite_md, gr.update(visible=True), initial_chat_history, initial_context, gr.Button(value="리포트 및 피드백 받기", interactive=True), gr.update(visible=True), gr.update(choices=final_sources, value=None)

def handle_chat_submission(question, history, initial_context, retriever):
    if retriever is None:
        raise gr.Error("오류: Retriever가 초기화되지 않았습니다.")
    
    history.append({"role": "user", "content": question})
    response_stream = get_follow_up(question, history, retriever, initial_context)
    
    bot_message = ""
    history.append({"role": "assistant", "content": ""})
    for chunk in response_stream:
        bot_message += chunk
        history[-1]["content"] = bot_message
        yield history, ""