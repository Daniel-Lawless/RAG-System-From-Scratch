# Progress Log

## Milestone 1 — Basic RAG pipeline

Implemented:
- Word-based chunking with overlap
- SentenceTransformer embeddings
- In-memory vector DB
- Top-k retrieval using cosine similarity
- OpenAI response generation using retrieved context

Key lesson:
- Retrieval quality controls answer quality. Increasing k from 2 to 4 allowed the model to retrieve the chunk containing the final weighted mean estimator.
- Normalizing vectors before calculating the cosine similarity reduces the computation down to only the dot product.

## Milestone 2 — Metadata-aware chunk records and retrieval debugging

Implemented:

- Replaced plain chunk storage with metadata-aware records
- Each vector DB record now stores:
  - chunk text
  - embedding
  - source file
  - chunk index
- Updated the RAG pipeline so each indexed chunk keeps track of where it came from
- Updated retrieval so search returns full records instead of just chunk text
- Added logging for indexing and retrieval debugging
- Added source file and chunk index information to the retrieved context passed to the model

Example retrieval debug output with query:

```text
What is the final probability for Buffon's needle?
```

![Metadata-aware retrieval debug output](../assets/retrieval-debug-output.png)

1: The system first indexes each markdown file in the data/ directory.


2: The debug logs then show the top-k chunks retrieved for the query.


3: Each retrieved chunk includes useful metadata:
- `similarity`: the cosine similarity between the query embedding and the chunk embedding
- `Retrieved chunk`: The chunk returned from the search algorithm
- `source_file`: the markdown file that the chunk came from
- `chunk_index`: the position of the chunk within that source file


4: The final answer is generated from the retrieved context.
```math
P(crossing) = \frac{2L}{\pi t}
```
This confirms that the system retrieved the correct context and used it to answer the query.

Key lesson:

- Metadata makes retrieval much easier to inspect and debug.
- Seeing the `source_file` and `chunk_index` confirms whether the system is retrieving context from the correct document.
- For the query asking about Buffon's Needle, the highest-ranked retrieved chunks came from `04-buffons-needle.md`, which is the expected behaviour.
- Some lower-ranked chunks may come from unrelated files when `k` is larger, so choosing an appropriate number of retrieved chunks matters.

## Milestone 3 — Recursive chunking and structure-aware splitting

Implemented:

- Replaced the original fixed-size word chunker with a recursive chunker
- The chunker now tries to split text using natural document structure:
  - paragraphs first
  - then sentences
  - then newlines
  - then fixed-size word chunks as a fallback
- Added overlap between neighbouring chunks so important context is not lost at chunk boundaries
- Adjusted the raw chunk size to `chunk_size - overlap` so that adding overlap does not push the final chunks above the target chunk size

Key lesson:

- Chunking quality has a direct effect on retrieval quality.
- Fixed-size word chunking is simple, but it can split important explanations, equations, or definitions at awkward points, causing the model to miss important context.
- Recursive chunking keeps related content together where possible by preferring paragraph and sentence boundaries before falling back to word-based splitting.
- Overlap helps preserve context between neighbouring chunks, especially when an explanation continues across a chunk boundary.

## Milestone 4 — Vectorized search with NumPy matrix multiplication

Implemented:

  * Replaced the previous heap-based search loop with vectorized NumPy search
  * Stored embeddings separately from records so they can be stacked into a matrix
  * Added an `embeddings_matrix` with shape:

    ```text
    (number_of_chunks, embedding_dimension)
    ```

  * Added a `_matrix_needs_rebuild` flag so the embeddings matrix is only rebuilt after new records are added
  * Used `np.vstack()` to stack all chunk embeddings into a single 2D matrix
  * Updated search to calculate all query-to-chunk similarities at once using matrix multiplication:

    ```python
    similarities = self.embeddings_matrix @ query
    ```

  * Used `np.argpartition()` to efficiently find the top-k most similar chunks without fully sorting every similarity score
  * Sorted only the selected top-k results before returning them from most similar to least similar

Key lesson:

  * By stacking all chunk embeddings into a matrix, retrieval can be written as one matrix-vector multiplication instead of a Python loop over every record. This makes the search
    implementation closer to how real vector databases for ML systems operate, by performing many vector comparisons together.
  * `np.argpartition()` is useful for top-k retrieval because it avoids fully sorting every similarity score when only the best few are needed.
  * Also, keeping the records and embeddings seperate makes the design cleaner and easier to debug. 
