import numpy as np
import logging
from typing import Any

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
