"""
Multi-RAG System - Main Application

A fullstack multimodal RAG (Retrieval-Augmented Generation) application
that supports text, images, and tables from PDFs.

Features:
- Keyword, semantic, and hybrid search
- Streaming and non-streaming responses
- Multiple Gemini model support
- Professional Gradio web interface
"""

import time
import os
import gradio as gr
from generation import (
    generate_rag_response, 
    validate_gemini_api_key
)
from helper import check_ollama_health, check_opensearch_health


# =====================================================================
#   HEALTH CHECK FUNCTIONS
# =====================================================================

def check_services():
    """Check if required services are running."""
    issues = []
    
    # Check OpenSearch
    if not check_opensearch_health():
        issues.append(
            f"OpenSearch ({os.getenv('OPENSEARCH_HOST', 'localhost')}:"
            f"{os.getenv('OPENSEARCH_PORT', '9200')})"
        )
    
    # Check Ollama
    if not check_ollama_health():
        issues.append(
            f"Ollama ({os.getenv('OLLAMA_HOST', 'localhost')}:"
            f"{os.getenv('OLLAMA_PORT', '11434')})"
        )
    
    if issues:
        return False, f"Services not available: {', '.join(issues)}"
    return True, "All services running"


# =====================================================================
#   MAIN PROCESSING FUNCTIONS
# =====================================================================

def process_query(query, search_type, model_type, stream, top_k):
    """Process and stream or return the final response."""
    if not query.strip():
        yield "⚠️ Please enter a question."
        return

    try:
        top_k = int(top_k)
        if stream:
            full_response = ""
            response_stream = generate_rag_response(
                query,
                search_type=search_type,
                top_k=top_k,
                stream=True,
                model_name=model_type,
            )
            if isinstance(response_stream, str):
                yield response_stream
                return
            for chunk in response_stream:
                full_response += chunk
                time.sleep(0.005)  # Slight delay for UI smoothness
                yield full_response
        else:
            result = generate_rag_response(
                query,
                search_type=search_type,
                top_k=top_k,
                model_name=model_type,
            )
            yield result

    except Exception as e:
        yield f"❌ Error: {str(e)}"


def ingest_uploaded_pdf(pdf_file, use_gemini_captions, reset_index):
    """Ingest a PDF selected from the Gradio UI."""
    if pdf_file is None:
        return "Please upload a PDF first."

    pdf_path = pdf_file if isinstance(pdf_file, str) else getattr(pdf_file, "name", None)
    if not pdf_path:
        return "Could not read uploaded PDF path."

    old_reset = os.environ.get("RESET_INDEX")
    os.environ["RESET_INDEX"] = "true" if reset_index else "false"

    try:
        from ingestion import ingest_pdf_file

        summary = ingest_pdf_file(pdf_path, use_gemini_captions=use_gemini_captions)
        return (
            "Ingestion complete.\n"
            f"Index: {summary['index_name']}\n"
            f"Text chunks: {summary['text_chunks']}\n"
            f"Tables: {summary['tables']}\n"
            f"Images: {summary['images']}"
        )
    except Exception as exc:
        return f"Ingestion failed: {exc}"
    finally:
        if old_reset is None:
            os.environ.pop("RESET_INDEX", None)
        else:
            os.environ["RESET_INDEX"] = old_reset


# =====================================================================
#   UI COMPONENTS
# =====================================================================

def create_ui():
    """Create the Gradio web interface."""
    css = """
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em !important;
        font-weight: bold;
    }
    .model-card:hover {
        box-shadow: 0 5px 15px rgba(102,126,234,0.3);
    }
    .command-grid {
        display: grid;
        gap: 12px;
        margin: 12px 0 18px;
    }
    .command-row {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 8px;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px;
        background: #fafafa;
    }
    .command-row code {
        white-space: pre-wrap;
        word-break: break-word;
        color: #111827;
    }
    .copy-command {
        border: 1px solid #d1d5db;
        background: white;
        border-radius: 6px;
        padding: 6px 10px;
        cursor: pointer;
    }
    """

    with gr.Blocks(title="LocalRAG Q&A System") as demo:
        
        # Inject CSS directly via HTML component
        gr.HTML(f"<style>{css}</style>")

        # Header
        gr.HTML("""
        <div style="text-align:center;margin-bottom:30px;">
            <h1 class="header-title">📚 LocalRAG Q&A System</h1>
            <p style="color:#666;font-size:1.1em;">
                Ask questions about your PDF documents and get accurate AI answers.
            </p>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("Live Demo"):
                with gr.Row():
                    # LEFT SIDE — INPUTS
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Your Question")
                        query_input = gr.Textbox(
                            placeholder="Ask anything about your PDF...",
                            lines=4,
                            container=False,
                            show_label=False
                        )

                        gr.Markdown("### ⚙️ Configuration")
                        with gr.Row():
                            with gr.Column():
                                search_type = gr.Radio(
                                    ["keyword", "semantic", "hybrid"],
                                    value="hybrid",
                                    label="🔍 Search Method"
                                )
                            with gr.Column():
                                model_type = gr.Radio(
                                    [
                                        ("Gemini 2.0 Flash", "gemini-2.0-flash"),
                                        ("Gemini 2.0 Flash Lite", "gemini-2.0-flash-lite"),
                                        ("Gemini 2.5 Pro", "gemini-2.5-pro"),
                                    ],
                                    value="gemini-2.0-flash",
                                    label="🤖 Model"
                                )

                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            stream_checkbox = gr.Checkbox(
                                label="🔄 Stream Response", 
                                value=True
                            )
                            top_k_slider = gr.Slider(
                                1, 10, value=5, step=1, 
                                label="📄 Documents to Retrieve"
                            )

                        submit_btn = gr.Button("✨ Generate Answer", variant="primary", scale=2)
                        clear_btn = gr.Button("🔄 Clear")

                    # RIGHT SIDE — OUTPUT
                    with gr.Column(scale=2):
                        gr.Markdown("### 💡 Answer")
                        output = gr.Textbox(
                            lines=25,
                            container=False,
                            interactive=False,
                            show_label=False
                        )

                        with gr.Row():
                            status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                            token_text = gr.Textbox(label="Characters", value="0", interactive=False)

                # Submit Logic
                def on_submit(query, search_type, model_type, stream, top_k):
                    if not query.strip():
                        yield "⚠️ Please enter a question.", "Error", "0"
                        return
                    
                    # Yield initial status
                    yield "", "Processing...", "0"
                    
                    full_text = ""
                    for chunk in process_query(query, search_type, model_type, stream, top_k):
                        full_text = chunk
                        # Yield update
                        yield full_text, "Generating...", str(len(full_text))
                    
                    # Final yield
                    yield full_text, "Done", str(len(full_text))

                # Clear Logic
                def clear_all():
                    return "", "", "Ready", "0"

                submit_btn.click(
                    on_submit,
                    inputs=[query_input, search_type, model_type, stream_checkbox, top_k_slider],
                    outputs=[output, status_text, token_text]
                )

                clear_btn.click(
                    clear_all,
                    outputs=[query_input, output, status_text, token_text]
                )

                # Example Questions
                gr.Markdown("### 📝 Example Questions")
                gr.Examples(
                    examples=[
                        ["How does RAG work?", "hybrid", "gemini-2.0-flash"],
                        ["Advantages of RAG vs fine-tuning?", "semantic", "gemini-2.5-pro"],
                        ["Explain retrieval steps in RAG", "keyword", "gemini-2.0-flash"],
                    ],
                    inputs=[query_input, search_type, model_type]
                )

            with gr.Tab("Setup Guide"):
                gr.Markdown("""
                ## 🛠️ Environment Setup
                
                ### Required Services
                1. **OpenSearch** - Vector database
                2. **Ollama** - Embeddings & text generation
                3. **Gemini API** - LLM responses
                """)
                
                gr.Markdown("### 🚀 Quick Start Commands")
                gr.HTML("""
                <div class="command-grid">
                    <div class="command-row">
                        <code>python -m venv .venv</code>
                        <button class="copy-command" onclick="navigator.clipboard.writeText('python -m venv .venv')">Copy</button>
                    </div>
                    <div class="command-row">
                        <code>source .venv/bin/activate</code>
                        <button class="copy-command" onclick="navigator.clipboard.writeText('source .venv/bin/activate')">Copy</button>
                    </div>
                    <div class="command-row">
                        <code>pip install -r requirements.txt</code>
                        <button class="copy-command" onclick="navigator.clipboard.writeText('pip install -r requirements.txt')">Copy</button>
                    </div>
                    <div class="command-row">
                        <code>docker compose up -d</code>
                        <button class="copy-command" onclick="navigator.clipboard.writeText('docker compose up -d')">Copy</button>
                    </div>
                    <div class="command-row">
                        <code>docker exec ollama ollama pull nomic-embed-text</code>
                        <button class="copy-command" onclick="navigator.clipboard.writeText('docker exec ollama ollama pull nomic-embed-text')">Copy</button>
                    </div>
                    <div class="command-row">
                        <code>python check_setup.py --services</code>
                        <button class="copy-command" onclick="navigator.clipboard.writeText('python check_setup.py --services')">Copy</button>
                    </div>
                    <div class="command-row">
                        <code>python ingestion.py</code>
                        <button class="copy-command" onclick="navigator.clipboard.writeText('python ingestion.py')">Copy</button>
                    </div>
                    <div class="command-row">
                        <code>python main.py</code>
                        <button class="copy-command" onclick="navigator.clipboard.writeText('python main.py')">Copy</button>
                    </div>
                </div>
                """)
                
                gr.Markdown("""
                ### 📝 Environment Variables
                
                Create `.env` in the project root:
                
                ```bash
                GEMINI_API_KEY=your_api_key_here
                ```
                
                Optional overrides (defaults shown):
                ```bash
                OPENSEARCH_HOST=localhost
                OPENSEARCH_PORT=9200
                OLLAMA_HOST=localhost
                OLLAMA_PORT=11434
                INDEX_NAME=pdf_content_index
                RERANK_RESULTS=true
                ```
                """)

            with gr.Tab("Ingest PDF"):
                gr.Markdown("""
                ## 📄 Index a PDF

                Upload a PDF, choose whether to use Gemini for image/table captions, then ingest it into OpenSearch.
                """)
                pdf_upload = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
                use_captioning = gr.Checkbox(label="Use Gemini captions for images and tables", value=True)
                reset_index = gr.Checkbox(label="Reset index before ingesting", value=False)
                ingest_btn = gr.Button("Index PDF", variant="primary")
                ingest_output = gr.Textbox(label="Ingestion Status", lines=8, interactive=False)

                ingest_btn.click(
                    ingest_uploaded_pdf,
                    inputs=[pdf_upload, use_captioning, reset_index],
                    outputs=[ingest_output],
                )

            with gr.Tab("Troubleshooting"):
                gr.Markdown("""
                ## ❓ Troubleshooting Guide
                
                ### Common Issues
                
                1. **Connection Errors**
                   - Check Docker: `docker ps`
                   - Restart OpenSearch: `docker compose restart opensearch`
                   - Restart Ollama: `docker compose restart ollama`
                
                2. **No Documents Found**
                   - Run ingestion: `python ingestion.py`
                   - Check index: `curl http://localhost:9200/pdf_content_index/_search`
                
                3. **API Key Errors**
                   - Verify `.env` file exists
                   - Check key at: https://aistudio.google.com/app/apikey
                
                4. **Import Errors**
                   - Reinstall deps: `pip install -r requirements.txt`
                """)

    return demo


# =====================================================================
#   RUN APP
# =====================================================================

if __name__ == "__main__":
    # Validate API key
    validate_gemini_api_key()
    
    # Create and launch UI
    app = create_ui()
    app.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=int(os.getenv("APP_PORT", "7860")),
        show_error=True
    )
