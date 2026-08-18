# 🏥 Medical RAG - Clinical Decision Support Engine

An evidence-grounded Retrieval-Augmented Generation (RAG) system for querying medical clinical practice guidelines (e.g., NICE Guidelines for Hypertension and Type 2 Diabetes) with exact section citations, page numbers, and visual grounding overlays.

---

## 🎯 Project Goal

The goal of this project is to provide healthcare professionals and researchers with precise, trustworthy clinical decision support by:
- **Layout-Aware Ingestion**: Extracting structure, section hierarchies, and visual bounding boxes from medical guideline PDFs using PaddleOCR and PyMuPDF.
- **Hybrid Retrieval**: Combining dense semantic vector search (ChromaDB + `BAAI/bge-small-en-v1.5`) and sparse keyword search (BM25) merged using Reciprocal Rank Fusion (RRF).
- **Grounded Generation**: Synthesizing accurate clinical answers strictly backed by retrieved guideline context with explicit section and page citations (e.g., `[Section 1.4.2, Page 15]`).
- **Visual Grounding UI**: Serving an interactive web interface that highlights evidence bounding boxes directly on original document page renders.

---

## 🚀 How to Run the Project

### 1. Prerequisites & Environment Setup

Ensure you have Python 3.10+ installed.

1. **Navigate to project directory:**
   ```bash
   cd AI-Hackathon
   ```

2. **Set up a Python Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your OpenRouter or OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   # OR
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

---

### 2. Running the Web Application (FastAPI + Visual UI)

Launch the end-to-end web server and clinical query interface:

```bash
python run_full_pipeline.py
```

- **Interactive UI**: Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Health Check**: Access [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### 3. Running Specific Pipelines (Optional CLI Tools)

- **Test Clinical Retrieval (CLI)**:
  Run hybrid retrieval queries directly in the terminal:
  ```bash
  python run_retrieval_pipeline.py --query "What is the first-line treatment for type 2 diabetes?" --mode hybrid
  ```

- **Run Document Ingestion Pipeline**:
  Parse medical PDFs and generate section-aware structured chunks:
  ```bash
  python run_paddle_pipeline.py
  ```

---

## 📁 Project Structure

```text
├── data/
│   ├── raw/          # Original PDF guideline documents
│   ├── ocr/          # Raw OCR markdown (.md) & layout json extractions
│   └── processed/    # Structured section JSON outputs for RAG engine
├── src/
│   ├── api/          # FastAPI REST endpoints & web UI server
│   ├── generation/   # LLM clinical answer synthesis & citation engine
│   ├── retrieval/    # Hybrid vector (ChromaDB) + BM25 RRF retriever
│   ├── ingestion/    # PDF parsers, PaddleOCR, & section-aware chunkers
│   └── ui/           # Visual Evidence Grounding HTML/JS UI panel
├── run_full_pipeline.py      # Main entry point to run web application
├── run_retrieval_pipeline.py # CLI runner for retrieval testing
├── run_paddle_pipeline.py    # Ingestion runner for PDF processing
├── requirements.txt          # Python project dependencies
└── README.md                 # Project documentation
```
