import json
import numpy as np

from index_manager import IndexManager


# FakeChunker and FakeEmbedder makes the test more isolated
class FakeChunker:
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 75) -> list[str]:
        # Split the text into chunks using "|" as a separator.
        # i.e., "alpha|beta" becomes ["alpha", "beta"].
        return text.split("|")
    
class FakeEmbedder:
    def embed(self, text: str) -> np.ndarray:
        # Return predictable fake embeddings for each chunk.
        if text == "alpha":
            return np.array([1.0, 0.0])
        if text == "beta":
            return np.array([0.0, 1.0])

        # Default embedding.
        return np.array([0.5, 0.5])


# Pytest creates a temporary folder just for this test.
# So we do not touch the real data/ or storage/index/ folders.
def test_index_manager_builds_records_and_embeddings(tmp_path):
    # Create fake data and index paths inside pytest's temporary folder.
    data_path = tmp_path / "data"
    index_path = tmp_path / "storage" / "index"

    # Create the temporary data directory.
    data_path.mkdir()

    # Create a fake markdown file.
    # FakeChunker will split this into two chunks: "alpha" and "beta".
    markdown_file = data_path / "test.md"
    markdown_file.write_text("alpha|beta", encoding="utf-8")

    # Create the IndexManager
    index_manager = IndexManager(
        embedder=FakeEmbedder(),  # type: ignore
        chunker=FakeChunker(),    # type: ignore
    )

    # Build the index from the fake data folder.
    # This should create records.jsonl and embeddings.npy in the folder.
    index_manager.build_index_from_data(
        data_path=data_path,
        index_path=index_path,
    )

    # Expected output files.
    records_path = index_path / "records.jsonl"
    embeddings_path = index_path / "embeddings.npy"

    # Check that both index files were created.
    assert records_path.exists()
    assert embeddings_path.exists()

    # Read records.jsonl back into Python dictionaries.
    with records_path.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]

    # Load the saved embedding matrix.
    embedding_matrix = np.load(embeddings_path)

    # We expect two chunks "alpha" and "beta".
    assert len(records) == 2

    # We expect two embeddings, each with dimension 2.
    assert embedding_matrix.shape == (2, 2)

    # Check the first saved record has the correct chunk text and metadata.
    assert records[0]["chunk"] == "alpha"
    assert records[0]["metadata"]["source_file"] == "test.md"
    assert records[0]["metadata"]["chunk_index"] == 0

    # Check the second saved record has the correct chunk text and metadata.
    assert records[1]["chunk"] == "beta"
    assert records[1]["metadata"]["source_file"] == "test.md"
    assert records[1]["metadata"]["chunk_index"] == 1

    # Check the saved embeddings match the fake embeddings returned by FakeEmbedder.
    np.testing.assert_array_equal(embedding_matrix[0], np.array([1.0, 0.0]))
    np.testing.assert_array_equal(embedding_matrix[1], np.array([0.0, 1.0]))