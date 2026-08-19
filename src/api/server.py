"""
FastAPI Server for Multi-Guideline Medical RAG Clinical Decision Support.
Exposes REST endpoints for clinical RAG queries and section hierarchy browsing across guidelines.
Serves interactive Visual Evidence Grounding UI Panel with bounding box overlay rendering.
"""

import json
import pathlib
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import config
from src.generation.generator import ClinicalRAGGenerator
from src.retrieval.retriever import ClinicalRetriever

app = FastAPI(
    title="Medical RAG Guideline Decision Support API",
    description="Evidence-Grounded Medical Guideline Decision Support API with Visual Grounding Panel",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registry of Available Guideline Generators
GENERATORS: Dict[str, ClinicalRAGGenerator] = {}


def resolve_json_path(filename: str) -> str:
    """Resolves JSON path, checking data/processed/ first then fallback to root."""
    processed_path = config.DATA_PROCESSED_DIR / filename
    if processed_path.exists():
        return str(processed_path)
    return filename


def get_generator(doc_key: str = config.DEFAULT_DOCUMENT_SLUG) -> ClinicalRAGGenerator:
    """Retrieves or initializes a ClinicalRAGGenerator for the specified document slug."""
    if doc_key in GENERATORS:
        return GENERATORS[doc_key]

    if doc_key == config.DEFAULT_DOCUMENT_SLUG or doc_key == "hypertension":
        json_path = resolve_json_path("hypertension_sections_output.json")
        coll_name = config.DEFAULT_COLLECTION_NAME
    elif doc_key == "diabetes" or doc_key == "type2_diabetes":
        json_path = resolve_json_path("paddle_sections_output.json")
        coll_name = "med_guidelines_BAAI_bge_small_en_v1_5"
    else:
        json_path = config.DEFAULT_PROCESSED_JSON_PATH
        coll_name = config.DEFAULT_COLLECTION_NAME

    retriever = ClinicalRetriever(
        json_chunks_path=json_path,
        collection_name=coll_name
    )
    retriever.initialize()
    gen = ClinicalRAGGenerator(retriever=retriever)
    GENERATORS[doc_key] = gen
    return gen


class QueryRequest(BaseModel):
    query: str
    document: Optional[str] = config.DEFAULT_DOCUMENT_SLUG
    top_k: int = config.DEFAULT_TOP_K
    mode: str = config.DEFAULT_SEARCH_MODE


@app.on_event("startup")
def startup_event():
    print("🚀 Pre-loading Default Hypertension & Diabetes Guideline Generators...")
    if pathlib.Path(resolve_json_path("hypertension_sections_output.json")).exists():
        get_generator("hypertension")
    if pathlib.Path(resolve_json_path("paddle_sections_output.json")).exists():
        get_generator("diabetes")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "loaded_documents": list(GENERATORS.keys())
    }


@app.post("/api/query")
def process_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    doc_key = req.document or "hypertension"
    gen = get_generator(doc_key)
    
    try:
        res = gen.generate(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode
        )
        res["document_slug"] = doc_key
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sections")
def get_sections(document: str = Query("hypertension")):
    if document == "hypertension":
        filename = "hypertension_sections_output.json"
    else:
        filename = "paddle_sections_output.json"

    path = pathlib.Path(resolve_json_path(filename))

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Section file for '{document}' not found.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "document_info": data.get("document_info", {}),
        "hierarchy_tree": data.get("hierarchy_tree", [])
    }


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    path = pathlib.Path("src/ui/index.html")
    if not path.exists():
        return HTMLResponse("<h2>Visual Evidence Panel UI under construction</h2>", status_code=200)
    
    with open(path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
