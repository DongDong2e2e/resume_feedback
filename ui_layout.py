# ui_layout.py

import gradio as gr
import ui_handlers

def create_ui(retriever_state, embeddings):
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
                
                with gr.Row():
                    output_report = gr.Markdown(label="사전 브리핑 리포트")
                    
                    with gr.Tabs():
                        with gr.TabItem("종합 분석"):
                            output_eval = gr.Markdown(label="역할별 평가 및 역량 스코어카드")
                        with gr.TabItem("항목별 피드백"):
                            output_itemized = gr.Markdown(label="항목별 상세 피드백")
                        with gr.TabItem("전체 수정 제안"):
                            output_rewrite = gr.Markdown(label="AI 추천 수정본")
                
                with gr.Accordion("AI가 참고한 자료 목록 (클릭하여 펼치기)", open=False):
                    source_documents_list = gr.Radio(
                        label="참고 문서 목록",
                        info="문서를 선택하면 아래에 전체 내용이 표시됩니다."
                    )
                    source_document_content = gr.Markdown(label="문서 전체 내용 보기")
                
                save_result_btn = gr.Button("결과 파일로 저장하기", visible=False)

                with gr.Column(visible=False) as chat_interface:
                    gr.Markdown("---")
                    gr.Markdown("### 추가 질문하기")
                    chatbot = gr.Chatbot(label="대화창", height=400, type='messages')
                    chat_input = gr.Textbox(label="질문", placeholder="피드백 내용에 대해 추가로 궁금한 점을 질문해보세요...")
                    chat_submit_btn = gr.Button("전송", variant="primary")
                
                submit_btn.click(
                    fn=ui_handlers.report_and_feedback_interface,
                    inputs=[company_input, job_title_input, jd_input, draft_input, retriever_state],
                    outputs=[output_report, output_eval, output_itemized, output_rewrite, chat_interface, chat_history, initial_context_state, submit_btn, save_result_btn, source_documents_list]
                )

                source_documents_list.change(
                    fn=ui_handlers.read_source_content_from_selection,
                    inputs=source_documents_list,
                    outputs=source_document_content
                )

                save_result_btn.click(
                    fn=ui_handlers.save_results_to_file,
                    inputs=[output_report, output_eval, output_itemized, output_rewrite, company_input, job_title_input],
                    outputs=None
                )
                
                chat_input.submit(
                    fn=ui_handlers.handle_chat_submission,
                    inputs=[chat_input, chat_history, initial_context_state, retriever_state], 
                    outputs=[chatbot, chat_input]
                )
                chat_submit_btn.click(
                    fn=ui_handlers.handle_chat_submission,
                    inputs=[chat_input, chat_history, initial_context_state, retriever_state], 
                    outputs=[chatbot, chat_input]
                )

            with gr.TabItem("자료 관리"):
                gr.Markdown("AI가 학습할 문서를 관리합니다. 파일을 업로드하면 AI가 분석하여 '지능형 분석 노트'를 자동으로 생성합니다.")
                
                status_box = gr.Textbox(label="상태", interactive=False)
                rebuild_retriever_btn = gr.Button("Retriever 다시 만들기", variant="stop")
                
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
                file_dropdown = gr.Dropdown(label="파일 선택", choices=ui_handlers.get_md_files(), interactive=True)
                file_content_box = gr.Textbox(label="파일 내용", lines=25, interactive=True)
                save_file_btn = gr.Button("수정된 내용 저장", variant="primary")

                rebuild_retriever_btn.click(
                    fn=lambda: ui_handlers.rebuild_retriever_for_ui(embeddings),
                    inputs=[], 
                    outputs=[status_box, retriever_state] 
                )

                process_file_btn.click(
                    fn=lambda files, context: ui_handlers.handle_file_upload(files, context, embeddings),
                    inputs=[file_upload_btn, file_context_input],
                    outputs=[status_box, file_dropdown, retriever_state] 
                )

                file_dropdown.change(fn=ui_handlers.read_file_content, inputs=file_dropdown, outputs=file_content_box)
                save_file_btn.click(fn=ui_handlers.manual_save_button_handler, inputs=[file_dropdown, file_content_box], outputs=[status_box, file_dropdown])
                
    return demo