"""
Centralized Configuration for Medical RAG Engine.
All parameters, file paths, models, token limits, retrieval weights, and server settings are managed here.
Modify these parameters to customize the behavior of the RAG system without modifying codebase logic.
"""

import os
import pathlib

# ==============================================================================
# 1. ROOT & DATA DIRECTORIES
# ==============================================================================
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_OCR_DIR = DATA_DIR / "ocr"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")

# Ensure required directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_OCR_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. DEFAULT DOCUMENT CONFIGURATION (HYPERTENSION GUIDELINE)
# ==============================================================================
DEFAULT_DOCUMENT_NAME = "Guideline for the pharmacological treatment of hypertension in adults"
DEFAULT_DOCUMENT_SLUG = "hypertension"
DEFAULT_SOURCE_URL = "https://www.who.int/publications/i/item/9789240033987"
DEFAULT_PAGE_OFFSET = 12  # Front-matter pages offset (Cover, TOC, Acknowledgements)

# Default File Paths for Primary Document
DEFAULT_RAW_PDF_PATH = str(DATA_RAW_DIR / "Guideline-for-the-pharmacological-treatment-of-hypertension-in-adults.pdf")
DEFAULT_OCR_JSON_PATH = str(DATA_OCR_DIR / "Guideline-for-the-pharmacological-treatment-of-hypertension-in-adults.json")
DEFAULT_OCR_MD_PATH = str(DATA_OCR_DIR / "Guideline-for-the-pharmacological-treatment-of-hypertension-in-adults.md")
DEFAULT_PROCESSED_JSON_PATH = str(DATA_PROCESSED_DIR / "hypertension_sections_output.json")


# ==============================================================================
# 3. VECTOR STORE & EMBEDDING CONFIGURATION
# ==============================================================================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Derived Collection Name for Vector Store
_model_slug = EMBEDDING_MODEL_NAME.replace('/', '_').replace('-', '_').replace('.', '_')
DEFAULT_COLLECTION_NAME = f"med_guidelines_{DEFAULT_DOCUMENT_SLUG}_{_model_slug}"


# ==============================================================================
# 4. SECTION CHUNKING CONFIGURATION
# ==============================================================================
MAX_CHUNK_TOKENS = 600
MIN_CHUNK_TOKENS = 30
CHUNK_OVERLAP_TOKENS = 100
TOKENIZER_MODEL = "cl100k_base"


# ==============================================================================
# 5. HYBRID RETRIEVAL & RRF CONFIGURATION
# ==============================================================================
DEFAULT_TOP_K = 4
DEFAULT_SEARCH_MODE = "hybrid"  # Options: 'hybrid', 'dense', 'bm25'
RRF_K_CONSTANT = 60
DENSE_SEARCH_WEIGHT = 0.6
SPARSE_SEARCH_WEIGHT = 0.4


# ==============================================================================
# 6. LLM & GENERATION CONFIGURATION
# ==============================================================================
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

LLM_FALLBACK_MODELS = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-4-31b-it:free"
]


# ==============================================================================
# 7. FASTAPI WEB SERVER CONFIGURATION
# ==============================================================================
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000


# ==============================================================================
# 8. EVALUATION CONFIGURATION
# ==============================================================================
EVAL_DATASET_PATH = str(DATA_DIR / "eval_dataset.json")
EVAL_RESULTS_PATH = str(DATA_DIR / "eval_results.json")
DEFAULT_EVAL_TOP_K = 4

