import pytest
from chunking import Chunking

def test_empty_text_returns_empty_list():
    chunker = Chunking()

    # If given an empty string, the chunker should return an empty list. 
    chunks = chunker.chunk_text("  ")

    # This means we expect chunks to be [], if it is not, fail.
    assert chunks == []

def test_chunk_overlap_cannot_equal_chunksize():
    chunker = Chunking()

    # This says it expects our code to raise a ValueError when these conditions are true.
    # If it does, the test passes, else it fails.
    with pytest.raises(ValueError):
        chunker.chunk_text("This is a test", chunk_size=10, overlap=10)

def test_chunk_overlap_cannot_equal_chunk_size():
    chunker = Chunking()

    with pytest.raises(ValueError):
        chunker.chunk_text("This is a test", chunk_size=10, overlap=11)

def test_short_text_returns_single_chunk():
    chunker = Chunking()

    chunks = chunker.chunk_text("This is a short test.", chunk_size=10, overlap=2)

    assert chunks == ["This is a short test."]

def test_overlap_cannot_be_negative():
    chunker = Chunking()

    with pytest.raises(ValueError):
        chunker.chunk_text("hello world", chunk_size=10, overlap=-1)

def test_overlap_is_added_to_later_chunks():
    chunker = Chunking()

    text = "one two three four five six seven eight nine ten"
    chunks = chunker.chunk_text(text, chunk_size=6, overlap=2)

    assert chunks[0] == "one two three four"
    assert chunks[1].startswith("three four")