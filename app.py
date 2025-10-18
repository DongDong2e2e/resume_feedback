import gradio as gr
import os
import sys
import json
import re
import glob
from dotenv import load_dotenv

# .env 파일로부터 환경변수를 로드합니다.
load_dotenv()

import nltk

from langchain_huggingface import HuggingFaceEmbeddings
# ======================================================================
# [수정] config에서 필요한 경로 설정들을 가져옵니다.
from config import BASE_DIR, DATA_DIRS, VECTOR_STORE_PATH, AI_GENERATED_NOTES_PATH, OLLAMA_BASE_URL
from data_loader import load_documents, split_documents
# [수정] create_and_save_vector_store와 load_vector_store를 임포트합니다.
from vector_store_manager import create_and_save_vector_store, load_vector_store
from orchestrator import generate_report_and_feedback, get_follow_up

# [수정] OLLAMA_HOST 환경 변수를 설정합니다. (OllamaEmbeddings 임포트 전에!)
# OLLAMA_HOST는 'http://' 없이 '호스트:포트' 형식이어야 합니다.
if OLLAMA_BASE_URL:
    # [수정] Gradio 최신 버전과의 호환성을 위해 http:// 스키마를 제거합니다.
    # 예: "http://127.0.0.1:11434" -> "127.0.0.1:11434"
    ollama_host = OLLAMA_BASE_URL.replace("http://", "").replace("https://", "")
    os.environ['OLLAMA_HOST'] = ollama_host
    print(f"Ollama 서버 주소를 '{os.environ['OLLAMA_HOST']}'로 설정합니다.")


from llm_handler import llm_gemini_flash_normal
from prompts import PDF_CONVERTER_PROMPT_TEMPLATE, TOPIC_SUGGESTION_PROMPT_TEMPLATE
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


print("전역 임베딩 모델로 HuggingFace 모델(Granite-Embedding-278m)을 생성합니다.")
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="ibm-granite/granite-embedding-278m-multilingual",
    model_kwargs={'device': 'cuda'},  # [수정] GPU를 사용하도록 'cuda'로 변경
    encode_kwargs={'normalize_embeddings': True}
)


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
setup_nltk()

# --- API 키 확인 로직 ---
if "GOOGLE_API_KEY" not in os.environ or os.environ["GOOGLE_API_KEY"] == "YOUR_API_KEY_HERE":
    print("오류: GOOGLE_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

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
def _rebuild_store_core():
    print("벡터 DB 재생성을 시작합니다...")
    # [수정] load_documents 함수는 이제 인자 없이 호출됩니다.
    all_documents = load_documents()
    if not all_documents:
        print("경고: 문서를 찾을 수 없어 벡터 DB를 생성할 수 없습니다.")
        return None 
    doc_chunks = split_documents(all_documents)
    vector_store = create_and_save_vector_store(doc_chunks, VECTOR_STORE_PATH, EMBEDDINGS)
    print("벡터 DB 재생성 완료.")
    return vector_store

def rebuild_vector_db_for_ui():
    """(복원) UI용 벡터 DB 새로고침 함수, 전역 변수 사용"""
    global VECTOR_STORE
    try:
        VECTOR_STORE = _rebuild_store_core()
        if VECTOR_STORE is None:
            return "벡터 DB 생성 중 오류가 발생했습니다. 터미널 로그를 확인해주세요."
        return "벡터 데이터베이스가 성공적으로 업데이트되었습니다."
    except Exception as e:
        print(f"벡터 DB 재생성 중 오류 발생: {e}")
        raise gr.Error(f"벡터 DB 업데이트 실패: {e}")

def initialize_vector_store():
    print("벡터 저장소 초기화를 시작합니다...")
    vector_store = load_vector_store(VECTOR_STORE_PATH, EMBEDDINGS)
    if vector_store is None:
        print(f"'{VECTOR_STORE_PATH}'에서 벡터 저장소를 찾을 수 없습니다. 새로 생성합니다.")
        vector_store = _rebuild_store_core()
    print("벡터 저장소 준비 완료.")
    return vector_store

VECTOR_STORE = initialize_vector_store()


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

def handle_file_upload(files, file_context, progress=gr.Progress(track_tqdm=True)):
    if not files:
        raise gr.Error("분석할 파일을 먼저 업로드해주세요.")

    processed_files = []
    total_files = len(files)
    
    if not file_context and total_files > 0:
        print("공통 주제가 없어 첫 파일 기준으로 AI 추천 주제를 사용합니다.")
        _, file_context = suggest_topic_from_file([files[0]])

    for i, file_obj in enumerate(files):
        try:
            progress(i / total_files, desc=f"({i+1}/{total_files}) '{os.path.basename(file_obj.name)}' 처리 중...")
            
            loader = UnstructuredFileLoader(file_obj.name)
            raw_text = loader.load()[0].page_content
            
            current_context = file_context
            if not current_context:
                suggestion_prompt = ChatPromptTemplate.from_template(TOPIC_SUGGESTION_PROMPT_TEMPLATE)
                suggestion_chain = suggestion_prompt | llm_gemini_flash_normal | StrOutputParser()
                current_context = suggestion_chain.invoke({"raw_text": raw_text})

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
    rebuild_status = rebuild_vector_db_for_ui()
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
    
    # 저장할 폴더 생성
    save_dir = "6_Consulting_Results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 파일명 생성
    today_str = datetime.now().strftime('%Y%m%d')
    safe_company = "".join(c for c in company if c.isalnum())
    safe_job = "".join(c for c in job_title if c.isalnum())
    filename = f"[{today_str}]_{safe_company}_{safe_job}_컨설팅결과.md"
    filepath = os.path.join(save_dir, filename)

    # 저장할 내용 조합
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

# --- [수정] Gradio 인터페이스 함수 ---
def report_and_feedback_interface(company_name, job_title, job_description, my_draft):
    if not all([company_name, job_title, job_description, my_draft]):
        raise gr.Error("모든 필드를 입력해주세요.")
    if not VECTOR_STORE:
        raise gr.Error("벡터 저장소가 비어있습니다. 자료를 추가하고 DB를 다시 만들어주세요.")

    # 4개의 출력 컴포넌트(리포트, 종합분석, 항목별, 전체수정)에 대한 초기 상태
    yield "...", "...", "...", "...", gr.update(visible=False), [], "", gr.Button(value="생성 중...", interactive=False), gr.update(visible=False)
    
    initial_context = f"회사명: {company_name}\n직무명: {job_title}"
    response_generator = generate_report_and_feedback(VECTOR_STORE, job_description, my_draft, company_name, job_title)

    final_report, final_feedback_json_str = "", ""
    for response_data in response_generator:
        final_report = response_data["report"]
        final_feedback_json_str = response_data["feedback"]
        # 스트리밍 중에는 원본 응답을 임시로 표시
        yield final_report, final_feedback_json_str, "...", "...", gr.update(visible=False), [], initial_context, gr.Button(value="생성 중...", interactive=False), gr.update(visible=False)

    # 최종 JSON을 파싱하여 3개의 Markdown으로 분리하는 로직
    overall_eval_md = "### 📊 종합 분석\n\n분석 내용을 생성하지 못했습니다."
    itemized_md = "### ✍️ 항목별 상세 피드백\n\n피드백을 생성하지 못했습니다."
    rewrite_md = "### ✨ AI 추천 수정본\n\n수정본을 생성하지 못했습니다."

    try:
        # LLM 응답에서 JSON 코드 블록만 안정적으로 추출
        match = re.search(r'```json\s*([\s\S]+?)\s*```', final_feedback_json_str)
        json_str = match.group(1) if match else final_feedback_json_str
        
        # [수정] 디버깅을 위해 최종 JSON 문자열을 터미널에 출력
        print("--- 최종 LLM 응답 (JSON 파싱 전) ---")
        print(json_str)
        print("-------------------------------------")

        data = json.loads(json_str)

        # [수정] .get()을 사용하고, 키가 없는 경우를 대비해 UI에 명확한 메시지 표시
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
    # 최종적으로 파싱된 3개의 Markdown을 각각의 출력 컴포넌트로 전달
    yield final_report, overall_eval_md, itemized_md, rewrite_md, gr.update(visible=True), initial_chat_history, initial_context, gr.Button(value="리포트 및 피드백 받기", interactive=True), gr.update(visible=True)

def handle_chat_submission(question, history, initial_context):
    if VECTOR_STORE is None:
        raise gr.Error("오류: 벡터 저장소가 초기화되지 않았습니다.")
    
    history.append({"role": "user", "content": question})
    response_stream = get_follow_up(question, history, VECTOR_STORE, initial_context)
    
    bot_message = ""
    history.append({"role": "assistant", "content": ""})
    for chunk in response_stream:
        bot_message += chunk
        history[-1]["content"] = bot_message
        yield history, ""

# --- Gradio UI 구성 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 나만의 AI 취업 컨설턴트")
    
    chat_history = gr.State([])
    initial_context_state = gr.State("")

    with gr.Tabs():
        with gr.TabItem("AI 컨설턴트"):
            gr.Markdown("과거 지원 데이터와 최신 정보를 종합하여, 지원서에 대한 '사전 브리핑 리포트'와 '상세 피드백'을 제공합니다.")
            with gr.Row():
                with gr.Column(scale=2):
                    jd_input = gr.Textbox(lines=20, label="채용 공고 (Job Description)")
                with gr.Column(scale=1):
                    company_input = gr.Textbox(label="회사명")
                    job_title_input = gr.Textbox(label="직무명")
                    draft_input = gr.Textbox(lines=11, label="자기소개서 초안")
                    submit_btn = gr.Button("리포트 및 피드백 받기", variant="primary")
            
            # [수정] 출력을 Row와 3개의 탭으로 구조화
            with gr.Row():
                output_report = gr.Markdown(label="사전 브리핑 리포트")
                
                with gr.Tabs():
                    with gr.TabItem("종합 분석"):
                        output_eval = gr.Markdown(label="역할별 평가 및 역량 스코어카드")
                    with gr.TabItem("항목별 피드백"):
                        output_itemized = gr.Markdown(label="항목별 상세 피드백")
                    with gr.TabItem("전체 수정 제안"):
                        output_rewrite = gr.Markdown(label="AI 추천 수정본")
            
            # [추가] 결과 저장 버튼
            save_result_btn = gr.Button("결과 파일로 저장하기", visible=False)

            with gr.Column(visible=False) as chat_interface:
                gr.Markdown("---")
                gr.Markdown("### 추가 질문하기")
                chatbot = gr.Chatbot(label="대화창", height=400, type='messages')
                chat_input = gr.Textbox(label="질문", placeholder="피드백 내용에 대해 추가로 궁금한 점을 질문해보세요...")
                chat_submit_btn = gr.Button("전송", variant="primary")
            
            # [수정] outputs 리스트에 새로 만든 Markdown 컴포넌트들을 연결
            submit_btn.click(
                fn=report_and_feedback_interface,
                inputs=[company_input, job_title_input, jd_input, draft_input],
                outputs=[output_report, output_eval, output_itemized, output_rewrite, chat_interface, chat_history, initial_context_state, submit_btn, save_result_btn]
            )

            # [추가] 저장 버튼 클릭 이벤트 핸들러
            save_result_btn.click(
                fn=save_results_to_file,
                inputs=[output_report, output_eval, output_itemized, output_rewrite, company_input, job_title_input],
                outputs=None
            )
            
            chat_input.submit(fn=handle_chat_submission, inputs=[chat_input, chat_history, initial_context_state], outputs=[chatbot, chat_input])
            chat_submit_btn.click(fn=handle_chat_submission, inputs=[chat_input, chat_history, initial_context_state], outputs=[chatbot, chat_input])

        with gr.TabItem("자료 관리"):
            # ... (자료 관리 탭 UI는 변경 없음)
            gr.Markdown("AI가 학습할 문서를 관리합니다. 파일을 업로드하면 AI가 분석하여 '지능형 분석 노트'를 자동으로 생성합니다.")
            
            status_box = gr.Textbox(label="상태", interactive=False)
            rebuild_db_btn = gr.Button("벡터 DB 다시 만들기", variant="stop")
            
            gr.Markdown("---")
            gr.Markdown("### AI 기반 자료 자동 정리")

            with gr.Row():
                file_upload_btn = gr.UploadButton(
                    "1. 파일 업로드 (PDF, DOCX, TXT 등)", file_count="multiple"
                )
                uploaded_file_display = gr.Textbox(label="선택된 파일", interactive=False)
            
            file_context_input = gr.Textbox(
                label="2. AI가 추천한 핵심 주제 (수정 가능)", 
                placeholder="파일을 업로드하면 AI가 자동으로 주제를 추천합니다."
            )
            process_file_btn = gr.Button("3. 업로드한 파일로 분석 노트 생성하기", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### 수동 파일 관리")
            file_dropdown = gr.Dropdown(label="파일 선택", choices=get_md_files(), interactive=True)
            file_content_box = gr.Textbox(label="파일 내용", lines=25, interactive=True)
            save_file_btn = gr.Button("수정된 내용 저장", variant="primary")

            rebuild_db_btn.click(fn=rebuild_vector_db_for_ui, inputs=[], outputs=status_box)
            file_upload_btn.upload(fn=suggest_topic_from_file, inputs=file_upload_btn, outputs=[uploaded_file_display, file_context_input])
            process_file_btn.click(fn=handle_file_upload, inputs=[file_upload_btn, file_context_input], outputs=[status_box, file_dropdown])
            file_dropdown.change(fn=read_file_content, inputs=file_dropdown, outputs=file_content_box)
            save_file_btn.click(fn=manual_save_button_handler, inputs=[file_dropdown, file_content_box], outputs=[status_box, file_dropdown])

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
    check_ollama_connection() # Gradio 앱 실행 전 연결 확인
    print("Gradio 앱을 시작합니다. 웹 브라우저에서 다음 주소로 접속하세요.")
    demo.launch()