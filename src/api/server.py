"""
FastAPI Server for Medical RAG Clinical Decision Support.
Exposes REST endpoints for clinical RAG queries and section hierarchy browsing.
Serves interactive Visual Evidence Grounding UI Panel with bounding box overlay rendering.
"""

import json
import pathlib
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.generation.generator import ClinicalRAGGenerator
from src.retrieval.retriever import ClinicalRetriever

app = FastAPI(
    title="NICE Guidelines Medical RAG API",
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

# Global Generator Instance
generator = ClinicalRAGGenerator()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    mode: str = "hybrid"
    section_filter: Optional[str] = None


@app.on_event("startup")
def startup_event():
    print("🚀 Initializing Medical RAG Clinical Retriever & Generator...")
    generator.retriever.initialize()


@app.get("/health")
def health_check():
    return {"status": "ok", "retriever_initialized": generator.retriever._initialized}


@app.post("/api/query")
def process_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        res = generator.generate(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sections")
def get_sections():
    path = pathlib.Path("paddle_sections_output.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="paddle_sections_output.json not found.")

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
