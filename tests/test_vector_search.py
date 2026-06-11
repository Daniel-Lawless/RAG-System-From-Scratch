import pytest
import numpy as np
from vector_search import VectorSearch

def test_empty_database_return_empty_list():
    vector_search = VectorSearch()

    query_embedding = np.array([1,2,2])

    results = vector_search.search(
        query_embedding=query_embedding,
        k=10
    )

    assert results == []

def test_vector_search_returns_most_similar_record_first():
    vector_search = VectorSearch()

    # Populate vector search with fake data.
    embedding1 = np.array([2, 5, 9])
    embedding2 = np.array([9, 3, 1])
    embedding3 = np.array([4, 5, 6])

    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    embedding3_norm = embedding3 / np.linalg.norm(embedding3)

    vector_search.add_record(
        chunk="This is chunk 1",
        embedding=embedding1_norm,
        source_file="03_buffons_needle.md",
        chunk_index=0,
    )

    vector_search.add_record(
        chunk="This is chunk 2",
        embedding=embedding2_norm,
        source_file="04_lighthouse_problem.md",
        chunk_index=1,
    )

    vector_search.add_record(
        chunk="This is chunk 3",
        embedding=embedding3_norm,
        source_file="05_mean_and_variance.md",
        chunk_index=2,
    )

    query_embedding = np.array([5, 7, 2])
    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)

    # 1st: chunk_index = 2   similarity ≈ 0.8645
    # 2nd: chunk_index = 1   similarity ≈ 0.8071
    # 3rd: chunk_index = 0   similarity ≈ 0.6801
    # Should return results in the order 2, 1, 0

    # Gather results
    results = vector_search.search(
        query_embedding=query_embedding_norm,
        k=3,
    )

    # Extract each chunk index
    chunk_indexes = [
        result["metadata"]["chunk_index"]
        for result in results
    ]

    # check it returned 3 results in the correct order.
    assert len(results) == 3
    assert chunk_indexes == [2, 1, 0]

# Should retrieve all results.
def test_vector_search_k_larger_than_records_returns_all_records():
    vector_search = VectorSearch()

    vector_search.add_record(
        chunk="chunk 1",
        embedding=np.array([1.0, 0.0]),
        source_file="one.md",
        chunk_index=0,
    )

    vector_search.add_record(
        chunk="chunk 2",
        embedding=np.array([0.0, 1.0]),
        source_file="two.md",
        chunk_index=1,
    )

    query_embedding = np.array([1.0, 0.0])

    # K is greater than the number of record in vector search
    results = vector_search.search(
        query_embedding=query_embedding,
        k=10,
    )

    # so it should return all records, 2 in this case. 
    assert len(results) == 2

def test_vector_search_k_less_than_or_equal_to_zero_returns_empty_list():
    vector_search = VectorSearch()

    vector_search.add_record(
        chunk="This is chunk 1",
        embedding=np.array([1.0, 0.0]),
        source_file="test.md",
        chunk_index=0,
    )

    query_embedding = np.array([1.0, 0.0])

    # User selects 0 record to be retrieved
    results = vector_search.search(
        query_embedding=query_embedding,
        k=0,
    )

    # So, it should return the empty list.
    assert results == []
