from keyword_search import KeywordSearch

# If it has not been populated, it should return the empty list.
def test_keyword_search_empty_index_returns_empty_list():
    keyword_search = KeywordSearch()

    results = keyword_search.search(query="buffon", k=4)

    assert results == []

# If k=0 then we should just return the empty list.
def test_keyword_search_k_less_than_or_equal_to_zero_returns_empty_list():
    keyword_search = KeywordSearch()

    keyword_search._add_chunk(
        chunk="Buffon's needle estimates pi.",
        metadata={"source_file": "buffons_needle.md", "chunk_index": 0},
    )

    results = keyword_search.search(query="buffon", k=0)

    assert results == []

def test_keyword_search_returns_matching_chunk():
    keyword_search = KeywordSearch()

    keyword_search._add_chunk(
        chunk="Buffon's needle is a probability problem.",
        metadata={"source_file": "buffons_needle.md", "chunk_index": 0},
    )

    keyword_search._add_chunk(
        chunk="The lighthouse problem uses a Cauchy distribution.",
        metadata={"source_file": "lighthouse.md", "chunk_index": 1},
    )

    # Since only the second chunk mentions lighthouse, it should have a higher score
    # than the first chunk, and thus should be returned before it. And if k=1, it should
    # be returned other the first chunk.
    results = keyword_search.search(query="lighthouse", k=1)

    assert len(results) == 1
    assert results[0]["metadata"]["source_file"] == "lighthouse.md"
    assert results[0]["retriever"] == "keyword"

def test_keyword_search_returns_empty_list_for_no_matches():
    keyword_search = KeywordSearch()

    keyword_search._add_chunk(
        chunk="Buffon's needle estimates pi.",
        metadata={"source_file": "buffons_needle.md", "chunk_index": 0},
    )

    # My implementation only returns results if their score is > 0, if no term
    # in the query matches, that chunk gets a score of 0, so no results should be
    # returned in this case.

    results = keyword_search.search(query="lighthouse", k=4)

    assert results == []

def test_keyword_search_empty_query_returns_empty_list():
    keyword_search = KeywordSearch()

    keyword_search._add_chunk(
        chunk="Buffon's needle estimates pi.",
        metadata={"source_file": "buffons_needle.md", "chunk_index": 0},
    )

    # 'the', 'and', 'of' are all stop words, and should have been filtered out.
    # this makes it an empty string and thus should return the empty list.
    results = keyword_search.search(query="the and of", k=4)

    assert results == []

def test_keyword_search_sorts_results_by_score():
    keyword_search = KeywordSearch()

    keyword_search._add_chunk(
        chunk="lambda",
        metadata={"source_file": "one_lambda.md", "chunk_index": 0},
    )

    keyword_search._add_chunk(
        chunk="lambda lambda lambda",
        metadata={"source_file": "three_lambdas.md", "chunk_index": 1},
    )

    # Chunks that contain a query term more frequently should generally 
    # be scored higher by BM25, so chunk 1 should be returned first,
    # then chunk 0
    results = keyword_search.search(query="lambda", k=2)

    source_files = [
        result["metadata"]["source_file"]
        for result in results
    ]

    chunk_index = [
        result["metadata"]["chunk_index"]
        for result in results
    ]

    assert source_files == ["three_lambdas.md", "one_lambda.md"]
    assert chunk_index == [1,0]