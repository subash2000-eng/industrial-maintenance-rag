"""
Phase 4 - Pydantic Schemas
Defines the shape of all API requests and responses.
Pydantic validates data automatically before it
reaches the RAG pipeline.
"""

from pydantic import BaseModel, Field
from typing  import Optional


# ════════════════════════════════════════════════════════
# REQUEST SCHEMAS — data coming INTO the API
# ════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """
    Request body for POST /query endpoint.
    Technician submits a fault description here.
    """
    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Fault description from the technician",
        json_schema_extra={
            "example": "Hydraulic pump making grinding noise"
        }
    )
    manual_filter: Optional[str] = Field(
        None,
        description="Restrict search to one specific manual"
    )
    top_k_retrieve: Optional[int] = Field(
        20,
        ge=5,
        le=50,
        description="Chunks to retrieve before re-ranking"
    )
    top_k_rerank: Optional[int] = Field(
        3,
        ge=1,
        le=10,
        description="Final chunks passed to LLM after re-ranking"
    )

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
    """
    Request body for POST /feedback endpoint.
    Technician rates the quality of an answer.
    """
    query_id: str = Field(
        ...,
        description="Query ID from the /query response"
    )
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1 (poor) to 5 (excellent)"
    )
    was_helpful: bool = Field(
        ...,
        description="Was the answer helpful?"
    )
    comment: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional comment from technician"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query_id"   : "q_abc123",
                "rating"     : 4,
                "was_helpful": True,
                "comment"    : "Very accurate repair steps"
            }
        }
    }


# ════════════════════════════════════════════════════════
# RESPONSE SCHEMAS — data going OUT of the API
# ════════════════════════════════════════════════════════

class SourceCitation(BaseModel):
    """
    A single source citation in the query response.
    Shows exactly which manual page was used.
    """
    source_num   : int
    manual_name  : str
    page_number  : int
    section_title: str
    confidence   : str
    rerank_score : float


class QueryResponse(BaseModel):
    query_id          : str
    query             : str
    answer            : str
    sources           : list[SourceCitation]
    confidence        : str
    model             : str
    chunks_used       : int
    response_time_s   : float
    timestamp         : str
    detected_language : str = "English"
    original_query    : str = ""


class ManualInfo(BaseModel):
    """
    Information about one ingested manual.
    """
    manual_name: str
    chunk_count: int


class ManualsResponse(BaseModel):
    """
    Response from GET /manuals endpoint.
    Lists all ingested manuals.
    """
    total_manuals: int
    total_chunks : int
    manuals      : list[ManualInfo]


class HealthResponse(BaseModel):
    """
    Response from GET /health endpoint.
    Shows system status at a glance.
    """
    status        : str
    ollama_running: bool
    chroma_chunks : int
    model         : str
    version       : str = "1.0.0"


class FeedbackResponse(BaseModel):
    """
    Response from POST /feedback endpoint.
    Confirms feedback was received.
    """
    status  : str
    query_id: str
    message : str


# ── Run this file directly to test ──────────────────────
if __name__ == "__main__":

    # Test QueryRequest validation
    print("🧪 Testing schema validation...\n")

    # Valid request — should work
    valid = QueryRequest(
        query          ="motor bearing overheating fault",
        top_k_retrieve =20,
        top_k_rerank   =3
    )
    print(f"✅ Valid request accepted")
    print(f"   Query: {valid.query}")
    print(f"   top_k_retrieve: {valid.top_k_retrieve}")
    print(f"   top_k_rerank  : {valid.top_k_rerank}")

    # Test short query — should fail
    print(f"\n🧪 Testing short query rejection...")
    try:
        invalid = QueryRequest(query="hi")
    except Exception as e:
        print(f"✅ Short query correctly rejected")
        print(f"   Reason: min_length=5 validation")

    # Test feedback schema
    feedback = FeedbackRequest(
        query_id   ="q_test123",
        rating     =4,
        was_helpful=True,
        comment    ="Steps were clear and accurate"
    )
    print(f"\n✅ Feedback schema works")
    print(f"   Rating : {feedback.rating}/5")
    print(f"   Helpful: {feedback.was_helpful}")

    print(f"\n✅ All schemas valid — no warnings")