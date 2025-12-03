from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict


from IA.query_data import process_query
from IA.red_contac import process_query_graph

router = APIRouter(
    prefix="/ai",
    tags=["ai"],
)

class QueryRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    debug: Optional[bool] = False

@router.post("/query")
def ai_query(payload: QueryRequest) -> Dict[str, Any]:
    if not payload.prompt:
        raise HTTPException(status_code=400, detail="El campo 'prompt' es requerido")

    resp = process_query(
        payload.prompt,
        session_id=payload.session_id,
        debug=payload.debug or False,
    )
    # process_query ya devuelve un dict con {answer, sql_query, ...}
    return resp


class GraphQuery(BaseModel):
    query_text: str
    debug: bool = False

@router.post("/graphs/assistant")
def graphs_assistant(body: GraphQuery):
    return process_query_graph(body.query_text, debug=body.debug)