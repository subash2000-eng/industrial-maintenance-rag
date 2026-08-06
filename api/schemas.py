from pydantic import BaseModel, Field
from typing   import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000)
    manual_filter  : Optional[str] = None
    top_k_retrieve : Optional[int] = Field(20, ge=5, le=50)
    top_k_rerank   : Optional[int] = Field(3, ge=1, le=10)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query"         : "Motor bearing overheating after 2 hours",
                "top_k_retrieve": 20,
                "top_k_rerank"  : 3,
                "manual_filter" : None
            }
        }
    }


class FeedbackRequest(BaseModel):
    query_id   : str
    rating     : int = Field(..., ge=1, le=5)
    was_helpful: bool
    comment    : Optional[str] = Field(None, max_length=500)


class SourceCitation(BaseModel):
    source_num   : int
    manual_name  : str
    page_number  : int
    section_title: str
    confidence   : str
    rerank_score : float


class QueryResponse(BaseModel):
    query_id         : str
    query            : str
    answer           : str
    sources          : list[SourceCitation]
    confidence       : str
    model            : str
    chunks_used      : int
    response_time_s  : float
    timestamp        : str
    detected_language: str = "English"
    original_query   : str = ""


class ManualInfo(BaseModel):
    manual_name: str
    chunk_count: int


class ManualsResponse(BaseModel):
    total_manuals: int
    total_chunks : int
    manuals      : list[ManualInfo]


class HealthResponse(BaseModel):
    status        : str
    ollama_running: bool
    chroma_chunks : int
    model         : str
    version       : str = "1.0.0"


class FeedbackResponse(BaseModel):
    status  : str
    query_id: str
    message : str