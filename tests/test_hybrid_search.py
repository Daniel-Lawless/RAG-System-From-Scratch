import pytest
import numpy as np

from vector_search import VectorSearch
from keyword_search import KeywordSearch
from hybrid_search import HybridSearch

# So pytest doesn't need to load the embedder model. 
# Keeps the tests isolated.
class FakeEmbedder:
    def embed(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0])
    
def test_hybrid_search_k_less_than_or_equal_to_zero_returns_empty_list():
    vector_search = VectorSearch()
    keyword_search = KeywordSearch()
    embedder = FakeEmbedder()

    hybrid_search = HybridSearch(
        vector_search=vector_search,
        keyword_search=keyword_search,
        embedder=embedder, # type: ignore
    )

    results = hybrid_search.search(query="test", k=0)

    assert results == []

def test_hybrid_search_negative_weight_raises_value_error():
    vector_search = VectorSearch()
    keyword_search = KeywordSearch()
    embedder = FakeEmbedder()

    hybrid_search = HybridSearch(
        vector_search=vector_search,
        keyword_search=keyword_search,
        embedder=embedder, # type: ignore
        vector_weight=-1,
        keyword_weight=1,
    )

    with pytest.raises(ValueError):
        hybrid_search.search(query="test", k=4)

def test_hybrid_search_both_weights_zero_raises_value_error():
    vector_search = VectorSearch()
    keyword_search = KeywordSearch()
    embedder = FakeEmbedder()

    hybrid_search = HybridSearch(
        vector_search=vector_search,
        keyword_search=keyword_search,
        embedder=embedder, # type: ignore
        vector_weight=0,
        keyword_weight=0,
    )

    with pytest.raises(ValueError):
        hybrid_search.search(query="test", k=4)

def test_hybrid_search_fuses_same_record_from_vector_and_keyword():
    vector_search = VectorSearch()
    keyword_search = KeywordSearch()
    embedder = FakeEmbedder()

    vector_search.add_record(
        chunk="lambda chunk",
        embedding=np.array([1.0, 0.0]),
        source_file="lambda.md",
        chunk_index=0,
    )

    keyword_search._add_chunk(
        chunk="lambda chunk",
        metadata={"source_file": "lambda.md", "chunk_index": 0},
    )

    # Two chunks with the same source_file and chunk_index
    # are the same chunk, so we should only keep one of them.

    hybrid_search = HybridSearch(
        vector_search=vector_search,
        keyword_search=keyword_search,
        embedder=embedder, # type: ignore
    )

    # This should only return one result.
    results = hybrid_search.search(query="lambda", k=4)

    assert len(results) == 1
    assert results[0]["metadata"]["source_file"] == "lambda.md"
    assert results[0]["metadata"]["chunk_index"] == 0
    assert results[0]["retriever"] == "hybrid(vector+keyword)"