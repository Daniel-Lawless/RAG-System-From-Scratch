import numpy as np
import logging
from pathlib import Path
from typing import Any
import json

logger = logging.getLogger(__name__)

class VectorSearch:

    def __init__(self):
        # Store the chunk text and the meta data
        self.records = []

        # Store the embeddings seperately so we can stack them into a matrix.
        self.embeddings = []

        # Stack embedding ontop of one another to form a matrix
        # Shape (number_of_chunks, embedding_dimension)
        self.embeddings_matrix = None

        # Tracks whether we need to rebuild the matrix after adding new records.
        self._matrix_needs_rebuild = True

    @classmethod
    def from_index(cls, index_path : Path = Path("storage/index")) -> "VectorSearch":
        # Define object to be returned.
        vector_search = cls()

        # File paths
        records_path = index_path / "records.jsonl"
        embeddings_path = index_path / "embeddings.npy"

        if not records_path.exists() or records_path.stat().st_size == 0:
            raise FileNotFoundError(f"Path {records_path} does not exist or is empty")

        if not embeddings_path.exists() or embeddings_path.stat().st_size == 0:
            raise FileNotFoundError(f"Path {embeddings_path} does not exist or is empty")

        logger.info("Loading records and embeddings...")

        # Convert each json "dict" in records.jsonl to a Python dict and add it to our records.  
        with records_path.open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                vector_search.records.append(record)
        
        # Load matrix from embeddings.npy
        loaded_matrix = np.load(embeddings_path)

        # If the file is empty
        if (loaded_matrix.size == 0):
            vector_search.embeddings_matrix = None
            vector_search.embeddings = []
        # else set matrix equal to the loaded matrix and populate embeddings.
        else:
            vector_search.embeddings_matrix = loaded_matrix
            vector_search.embeddings = [row for row in vector_search.embeddings_matrix]
        
        if len(vector_search.records) != len(vector_search.embeddings):
            raise ValueError(
                "Number of records does not match number of embeddings"
            )

        # Matrix is already loaded, so no need to rebuild
        vector_search._matrix_needs_rebuild = False

        logger.info("Records and embeddings loaded.")

        return vector_search

    # Adds a record to the vector database
    def add_record(self, chunk: str, embedding: np.ndarray, source_file: str, chunk_index: int) -> None:

        # Construct record
        record = {
            "chunk" : chunk,
            "metadata" : {
                "source_file" : source_file,
                "chunk_index" : chunk_index
            }
        }

        # Append record to our list of records
        self.records.append(record)

        # Append the chunks embedding to our list of embeddings.
        self.embeddings.append(embedding)

        # A new embedding was added, so the matrix is out of date. Hence, it needs to be rebuilt. 
        self._matrix_needs_rebuild = True

    def _rebuild_embedding_matrix(self) -> None:

        # If the matrix does not need to be rebuilt, just return.
        if not self._matrix_needs_rebuild:
            return

        # If there are no embeddings, there is nothing to build.
        if not self.embeddings:
            self.embeddings_matrix = None
            self._matrix_needs_rebuild = False
            return

        # np.vstack() takes a list of vectors and stacks them on top of each other to form a 2d matrix.
        self.embeddings_matrix = np.vstack(self.embeddings)

        # The matrix is now up to date.
        self._matrix_needs_rebuild = False

    # Returns the top k most similar records in our vector database to the user query.
    def search(self, query_embedding: np.ndarray, k: int) -> list[dict[str, Any]]:

        logger.info("Starting vector search...")
        
        logger.info("Fetching most similar chunks...")

        results = []

        # If k is invalid or the database is empty, return no results.
        if k <= 0 or not self.records:
            return []
        
        # Check if the embeddings matrix needs rebuilding, if it does, rebuild it. 
        self._rebuild_embedding_matrix()

        # This means no embeddings were present.
        if self.embeddings_matrix is None:
            return []

        # If k is bigger than the number of records, retrieve all records.
        k = min(k, len(self.records))

        # Now, we calculate the dot product between all embeddings with the query at once. This is much more efficient
        # than using a heap. This will give a numpy.ndarray of shape (num_chunks,), one similarity for each chunk.
        similarities = self.embeddings_matrix @ query_embedding

        # Returns the indices of the top k scores. argpartition is faster than sorting the entire array.
        top_k_indices = np.argpartition(similarities, -k)[-k:]

        # Since argpartition does not return them in sorted order, now sort only the top k results.
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

        for index in top_k_indices:
            index = int(index)
            record = self.records[index]

            results.append({
                "chunk": record["chunk"],
                "score": float(similarities[index]),
                "retriever": "vector",
                "metadata": record["metadata"],
            })

        logger.info("Most similar chunks fetched")

        logger.info("Vector search complete")

        return results
