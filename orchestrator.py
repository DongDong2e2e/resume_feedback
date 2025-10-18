# --- START OF FILE orchestrator.py ---

import os
from datetime import datetime
import time
import itertools
from langchain_core.prompts import ChatPromptTemplate
from ddgs import DDGS

# [핵심] 분리된 모듈에서 프롬프트와 LLM 핸들러를 가져옵니다.
from prompts import *
from llm_handler import invoke_with_fallback, ALL_LLMS # ALL_LLMS 임포트
from config import NORMAL_MODELS_FALLBACK_ORDER, STREAM_MODELS_FALLBACK_ORDER # config에서 목록 임포트


WEB_SEARCH_CACHE = {}

# --- [수정] 프롬프트 객체를 파일 최상단에서 한 번만 생성 ---
briefing_prompt = ChatPromptTemplate.from_template(BRIEFING_PROMPT_TEMPLATE)
research_prompt = ChatPromptTemplate.from_template(RESEARCH_PROMPT_TEMPLATE)
summary_prompt = ChatPromptTemplate.from_template(RESEARCH_SUMMARY_PROMPT_TEMPLATE)
# draft_prompt = ChatPromptTemplate.from_template(DRAFT_PROMPT_TEMPLATE) # Chained Prompt로 대체
critique_prompt = ChatPromptTemplate.from_template(CRITIQUE_PROMPT_TEMPLATE)
resolver_prompt = ChatPromptTemplate.from_template(RESOLVER_PROMPT_TEMPLATE)
conversational_prompt = ChatPromptTemplate.from_template(CONVERSATIONAL_PROMPT_TEMPLATE)

# Chained Prompts
analysis_prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT_TEMPLATE)
evaluation_prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT_TEMPLATE)
synthesis_prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT_TEMPLATE)


# --- 메인 오케스트레이터 함수 ---
def generate_report_and_feedback(vector_store, jd_text, draft_text, company_name, job_title):
    """
    RAG 검색 -> 브리핑 리포트 생성 -> 웹 검색 -> 최종 피드백 생성을 순차적으로 수행하는 제너레이터
    """
    print("\n[Orchestrator] 리포트 및 피드백 생성 프로세스 시작.")
    start_time = time.time()
    
    # config에서 가져온 이름 목록을 실제 llm 인스턴스로 변환
    normal_models = [ALL_LLMS[name] for name in NORMAL_MODELS_FALLBACK_ORDER]
    stream_models = [ALL_LLMS[name] for name in STREAM_MODELS_FALLBACK_ORDER]

    report = ""
    try:
        # --- 1단계: 내부 데이터(RAG) 수집 ---
        yield {"report": "### ⏳ 1/4: 관련 내부 자료 검색 중...", "feedback": "", "done": False}
        retriever = vector_store.as_retriever(search_kwargs={'k': 7})
        retrieval_query = f"{company_name} {job_title} 직무 지원"
        relevant_docs = retriever.invoke(retrieval_query)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    except Exception as e:
        yield {"report": f"### ❌ 오류 (1단계: 내부 자료 검색)\n\n{e}", "feedback": "", "done": True}
        return

    try:
        # --- 2단계: 사전 브리핑 리포트 생성 ---
        yield {"report": "### ⏳ 2/4: 사전 브리핑 리포트 생성 중...", "feedback": "", "done": False}
        briefing_inputs = {"company": company_name, "job_title": job_title, "context": context_text}
        report_models = [llm_gemini_flash_normal, llm_ollama_normal]
        report, report_model = invoke_with_fallback(briefing_prompt, briefing_inputs, report_models)
    except Exception as e:
        yield {"report": f"### ❌ 오류 (2단계: 리포트 생성)\n\n{e}", "feedback": "", "done": True}
        return

    try:
        # --- 3단계: 웹 리서치 및 최종 피드백 준비 ---
        feedback = "### ⏳ 3/4: 최신 외부 정보 검색 및 요약 중..."
        yield {"report": f"✅ **사전 브리핑 리포트 (by {report_model})**\n\n---\n{report}", "feedback": feedback, "done": False}
        
        research_inputs = {"company": company_name, "job_title": job_title, "jd": jd_text}
        raw_keyword_output, _ = invoke_with_fallback(research_prompt, research_inputs, normal_models)
        
        keyword_phrases = [line.strip() for line in raw_keyword_output.strip().splitlines() if line.strip()]

        search_queries = []
        if len(keyword_phrases) > 0:
            search_queries.append(f"{company_name} {keyword_phrases[0]}")
        if len(keyword_phrases) > 1:
            search_queries.append(f"{job_title} {keyword_phrases[1]}")

        if not search_queries:
            print("  - 경고: LLM이 유효한 검색 키워드를 생성하지 못했습니다. 웹 검색을 건너뜁니다.")
            web_results = "생성된 검색어가 없어 웹 검색을 건너뛰었습니다."
        else:
            web_results = ""
            with DDGS() as ddgs:
                for query in search_queries:
                    print(f'  - 웹 검색 실행: \"{query}\"')
                    search_results_list = ddgs.text(query, max_results=3)
                    result_str = "\n".join([f"- {r['title']}: {r['body']}" for r in search_results_list])
                    web_results += f"[검색 쿼리: {query}]\n{result_str}\n\n"
        
        save_search_results_to_markdown(company_name, job_title, web_results)
        
        summary_inputs = {"web_results": web_results}
        research_summary, _ = invoke_with_fallback(summary_prompt, summary_inputs, [llm_ollama_normal])
    except Exception as e:
        yield {"report": f"### ❌ 오류 (3단계: 웹 리서치)\n\n{e}", "feedback": "", "done": True}
        return

    try:
        # --- 4단계: 최종 피드백 스트리밍 (Chained Prompt & MCP) ---
        feedback = "### ⏳ 4/4: 최종 피드백 생성 및 자가 교정 중..."
        yield {"report": f"✅ **사전 브리핑 리포트 (by {report_model})**\n\n---\n{report}", "feedback": feedback, "done": False}

        # 1. 분석 및 근거 추출
        analysis_inputs = {"context": context_text, "draft": draft_text}
        analysis_report, _ = invoke_with_fallback(analysis_prompt, analysis_inputs, normal_models)

        # 2. 다중 관점 평가
        evaluation_inputs = {
            "analysis_report": analysis_report, 
            "research_summary": research_summary, 
            "jd": jd_text
        }
        evaluation_report, _ = invoke_with_fallback(evaluation_prompt, evaluation_inputs, normal_models)

        # 3. 종합 및 수정 제안 (이 단계의 결과가 '피드백 초안'이 됨)
        synthesis_inputs = {
            "analysis_report": analysis_report,
            "evaluation_report": evaluation_report,
            "draft": draft_text
        }
        # 'synthesis_report'는 이제 JSON 형식의 문자열 초안입니다.
        draft_feedback_json, draft_model = invoke_with_fallback(synthesis_prompt, synthesis_inputs, normal_models)

        # 4. 품질 검수 및 최종본 생성
        # [수정] draft_feedback_json을 critique_prompt에 직접 전달
        critique, _ = invoke_with_fallback(critique_prompt, {"draft_feedback_json": draft_feedback_json}, normal_models)

        final_model = draft_model

        if critique.strip() == "문제 없음":
            print(f"\n[MCP] 초안이 완벽하여, '{draft_model}' 모델의 결과로 바로 스트리밍합니다.")
            # [수정] 이미 완성된 JSON이므로 그대로 스트리밍
            resolver_stream = iter(draft_feedback_json) 
        else:
            print("\n[MCP] 비판 내용을 바탕으로 최종본 생성 및 스트리밍 시작...")
            # [수정] resolver_prompt에 draft_feedback 대신 draft_feedback_json을 전달
            resolver_inputs = {
                "original_request": f"Analysis: {analysis_report}\n\nEvaluation: {evaluation_report}",
                "draft_feedback_json": draft_feedback_json, 
                "critique": critique
            }
            resolver_stream, final_model = invoke_with_fallback(resolver_prompt, resolver_inputs, stream_models, stream=True)

        full_feedback_response = ""
        for chunk in resolver_stream:
            full_feedback_response += chunk
            yield {"report": f"✅ **사전 브리핑 리포트 (by {report_model})**\n\n---\n{report}", "feedback": full_feedback_response, "done": False}

        end_time = time.time()
        duration = end_time - start_time
        final_info = f"\n\n---\n**모델:** {final_model} | **소요 시간:** {duration:.2f}초"
        yield {"report": f"✅ **사전 브리핑 리포트 (by {report_model})**\n\n---\n{report}", "feedback": full_feedback_response + final_info, "done": True}
    except Exception as e:
        error_message = f"\n\n---\n**오류:** 피드백 생성 중 심각한 오류가 발생했습니다. - {e}"
        yield {"report": report, "feedback": error_message, "done": True}


# --- 후속 질문 처리 함수 ---
def get_follow_up(question, chat_history, vector_store, initial_context):
    """대화 기록과 초기 컨텍스트를 바탕으로 후속 질문에 답변합니다."""
    print(f"\n[Follow-up Handler] 후속 질문 처리 시작: \"{question}\"\n")
    
    # 1. 대화 기록을 문자열로 변환
    history_str = ""
    for message in chat_history:
        role = "사용자" if message["role"] == "user" else "AI"
        history_str += f"{role}: {message['content']}\n"

    # 2. 새로운 질문과 대화 맥락을 기반으로 관련 문서 다시 검색
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    retrieval_query = f"초기 컨텍스트: {initial_context}\n\n대화 기록: {history_str}\n\n사용자의 질문: {question}"
    relevant_docs = retriever.invoke(retrieval_query)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 3. 프롬프트에 필요한 입력값 준비
    inputs = {
        "chat_history": history_str,
        "context": context_text,
        "question": question
    }

    # 4. 스트리밍 폴백 모델로 답변 생성
    stream_models = [ALL_LLMS[name] for name in STREAM_MODELS_FALLBACK_ORDER]
    
    try:
        response_stream, model_name = invoke_with_fallback(conversational_prompt, inputs, stream_models, stream=True)
        print(f"  - 후속 질문 답변 생성 중 (by {model_name})...")
        
        # 생성된 응답을 스트리밍으로 반환
        yield from response_stream

    except Exception as e:
        print(f"\n[Follow-up Handler] 후속 질문 처리 중 오류 발생: {e}")
        yield f"\n\n---\n**오류:** 답변 생성 중 오류가 발생했습니다. - {e}"

# --- 헬퍼 함수 ---
def save_search_results_to_markdown(company_name, job_title, web_results):
    """웹 검색 결과를 체계적인 마크다운 파일로 저장합니다."""
    save_dir = "5_Web_Search_Cache"
    os.makedirs(save_dir, exist_ok=True)

    today_str = datetime.now().strftime('%Y%m%d')
    safe_company_name = "".join(c if c.isalnum() else "_" for c in company_name)
    safe_job_title = "".join(c if c.isalnum() else "_" for c in job_title)
    filename = f"[웹검색]_{safe_company_name}_{safe_job_title}_{today_str}.md"
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath):
        print(f"  ℹ️ 오늘 날짜의 웹 검색 결과 파일이 이미 존재하여 저장을 건너뜁니다: {filepath}")
        return

    content = f"# {company_name} - {job_title} 직무 관련 웹 검색 결과\n\n"
    content += f"- **검색 일자:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    content += f"- **관련 공고:** {job_title}\n"
    content += f"- **대상 기업:** {company_name}\n\n"
    content += "---\n\n"
    content += "## 검색 결과 원본\n\n"
    content += f'''```text
{web_results}
```'''

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ 웹 검색 결과가 성공적으로 저장되었습니다: {filepath}")
    except Exception as e:
        print(f"  ❌ 웹 검색 결과 저장 중 오류 발생: {e}")