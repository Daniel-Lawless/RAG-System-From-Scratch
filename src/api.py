from typing import Literal
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from embeddings import Embeddings
from vector_search import VectorSearch
from keyword_search import KeywordSearch
from hybrid_search import HybridSearch
from rag_pipeline import RAGPipeline 
from index_loader import prepare_index

# Create the API application
app = FastAPI(title="Mini RAG system API")

# Get where the index was saved to.
INDEX_PATH = prepare_index()

# initialising these objects only happens at startup time.
# They do not get reloaded with each request.
embedder = Embeddings()
vector_search = VectorSearch.from_index(index_path=INDEX_PATH)
keyword_search = KeywordSearch.from_jsonl(index_path=INDEX_PATH)
hybrid_search = HybridSearch(
    vector_search=vector_search,
    keyword_search=keyword_search,
    embedder=embedder
)
rag_pipeline = RAGPipeline()

# Defines the expected JSON body for /query
class QueryRequest(BaseModel):
    query: str
    retriever: Literal["vector", "keyword", "hybrid"] = "hybrid"
    # Prevents invalid k
    k: int = Field(default=4, ge=1, le=20)

# Defines the shape of each retrieved source in results.
class RetrievedSource(BaseModel):
    source_file: str
    chunk_index: int
    score: float
    retriever_name: str

# Defines the shape of the full response.
class QueryResponse(BaseModel):
    answer: str
    retrieved_sources: list[RetrievedSource]

# Is the API alive endpoint.
@app.get("/")
def health_check() -> dict[str, str]:
    return {"status" : "ok"}

# Create a query endpoint
# In the FastAPI docs, it says the response_model is used for documentation,
# validation, and conversion
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    # Because QueryRequest is a pydantic model, FastAPI reads the incoming
    # JSON deserialises it into a QueryRequest object.  This is why we can
    # access its fields like normal Python.

    # Run specificed retriever
    if request.retriever == "vector":

        # Embed the query
        query_embedding = embedder.embed(request.query)

        # Extract results from vector search
        results = vector_search.search(
            query_embedding=query_embedding,
            k=request.k
        )

    elif request.retriever == "keyword":
        # Extract results from keyword search
        results = keyword_search.search(
            query=request.query,
            k=request.k
        )

    elif request.retriever == "hybrid":
        # Extract results from hybrid search
        results = hybrid_search.search(
            query=request.query,
            k=request.k
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown retriever: {request.retriever}"
        )

    # Generate answer from the context retrieved by the chosen retriever
    answer = rag_pipeline.response(
        query=request.query,
        results=results
    )

    # Returns a list of pydantic objects
    retrieved_sources = [
        RetrievedSource(
            source_file=result["metadata"]["source_file"],
            chunk_index=result["metadata"]["chunk_index"],
            score=round(result["score"], 4),
            retriever_name=result["retriever"]
        )
        for result in results
    ]

    # FastAPI will take this Python object and turn it into JSON for
    # the HTTP response. It will serialise the Pydantic objects into JSON.
    return QueryResponse(
        answer=answer,
        retrieved_sources=retrieved_sources
    )
