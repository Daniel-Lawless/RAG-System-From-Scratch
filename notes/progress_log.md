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

## Milestone 5 — Persistent index storage and index management

Implemented:

* Added persistent index storage so the system no longer needs to rebuild the index every time it runs
* Created an `IndexManager` to handle indexing, saving, and loading
* Saved chunk records to:

```text
storage/index/records.jsonl
```

* Saved embeddings separately to:

```text
storage/index/embeddings.npy
```

* Updated the retrieval system so records and embeddings can be loaded back into memory from disk
* Added support for rebuilding the vector search object from the saved index

Key lesson:

* Persisting the index makes the project much closer to a real RAG system because indexing and querying become separate stages.
* Saving metadata and embeddings separately keeps the design clean:
  * metadata remains human-readable in JSONL
  * embeddings are stored efficiently as a NumPy array
* This also makes the system easier to deploy because the API can load an existing index rather than generating one on startup.

## Milestone 6 — Command-line interface and automatic indexing

Implemented:

* Added a command-line interface for asking questions from the terminal
* Added support for choosing the retrieval strategy from the CLI
* Added arguments for:
  * query
  * number of retrieved chunks
  * retriever type
  * vector and keyword weights for hybrid search
* Added automatic index checking so the CLI can build the index if it does not already exist
* Removed the need for a separate manual indexing step during normal usage

Example command:

```bash
python cli.py ask "Why does the lighthouse problem produce a Cauchy distribution?" --retriever hybrid -k 4
```

Key lesson:

* A CLI makes the system much easier to test and demonstrate.
* Automatically checking whether the index exists improves usability because the user does not need to remember the exact setup order.
* Exposing retrieval options through CLI arguments makes the system easier to experiment with.

## Milestone 7 — BM25 keyword search

Implemented:

* Added a keyword-based retriever using the BM25 ranking algorithm
* Tokenised chunks into searchable terms
* Stored:
  * term frequencies
  * document frequencies
  * document lengths
  * average document length
* Added BM25 scoring using `k1` and `b` parameters
* Added support for populating keyword search from the saved JSONL records
* Added keyword search support in the CLI

Key lesson:

* Vector search is good for semantic similarity, but it can miss exact technical terms.
* Keyword search is useful when the user query contains exact terms like formulas, the name of problems, or algorithm names.

## Milestone 8 — Hybrid retrieval with weighted Reciprocal Rank Fusion

Implemented:

* Added hybrid search to combine vector search and keyword search
* Used weighted Reciprocal Rank Fusion to merge ranked results from multiple retrievers
* Added adjustable vector and keyword weights
* Stored per-retriever scores so it is possible to inspect whether a chunk was found by:
  * vector search
  * keyword search
  * both retrievers
* Added hybrid retrieval as a CLI option

Key lesson:

* Hybrid retrieval is more robust than either vector search or keyword search alone.
* Vector search captures semantic similarity.
* Keyword search captures exact term matches.
* Weighted Reciprocal Rank Fusion is a way to combine retrievers because it uses rank positions rather than requiring raw scores from different systems to be directly comparable, so we do not have to normalize scores to compare them.

## Milestone 9 — Retrieval evaluation

Implemented:

* Created a retrieval evaluation pipeline
* Added an evaluation dataset of question and relevant-source pairs
* Evaluated multiple retrievers:
  * vector
  * keyword
  * hybrid
* Added retrieval metrics:
  * `Hit@k`
  * `Mean Reciprocal Rank`
  * `Correct_source_count@k`
  * `Precision`
* Saved evaluation for each retriever in `results.json` and results per question per retriever in `per_question_results.json`
* Optimised evaluation so retrievers and embeddings are loaded once and reused across all questions

Key lesson:

* Evaluation makes retrieval quality measurable instead of relying only on manual inspection.
* `Hit@k` shows whether the correct document appears anywhere in the top results.
* `MRR` rewards systems that rank the first relevant result highly.
* `Correct_source_count@k` tells us out of k retrieved results, how many were taken from the correct source
* `Precision@k` tells us the percentage of results that had the correct sourcefile. 
* Loading the index once rather than rebuilding it for every question made evaluation significantly faster.

## Milestone 10 — Testing and continuous integration

Implemented:

* Added a Pytest test suite for the main components of the RAG system
* Tested chunking behaviour, including:
  * empty text returning an empty list
  * short text returning a single chunk
  * invalid overlap values raising `ValueError`
  * negative overlap raising `ValueError`
  * overlap being correctly added to later chunks
* Tested vector search behaviour, including:
  * empty indexes returning an empty list
  * cosine similarity ranking returning the most similar records first
  * `k` values larger than the number of records returning all records
  * `k <= 0` returning an empty list
* Tested BM25 keyword search behaviour, including:
  * empty indexes returning an empty list
  * matching queries returning the expected chunk
  * queries with no matching terms returning an empty list
  * stop-word-only queries returning an empty list
  * higher-frequency keyword matches being ranked above lower-frequency matches
* Tested hybrid search behaviour, including:
  * `k <= 0` returning an empty list
  * negative retriever weights raising `ValueError`
  * both retriever weights being zero raising `ValueError`
  * duplicate chunks from vector and keyword search being fused into one hybrid result
* Tested index creation using fake chunkers and fake embedders
* Verified that the `IndexManager` creates:
  * `records.jsonl`
  * `embeddings.npy`
* Verified that saved records contain the correct:
  * chunk text
  * source file
  * chunk index
* Verified that saved embeddings have the expected shape and values
* Added a GitHub Actions CI workflow
* Configured CI to:
  * run on push
  * run on pull request
  * install dependencies
  * run the Pytest suite

Key lesson:

* Tests make the project safer to refactor because retrieval, chunking, hybrid search, and indexing behaviour can be checked automatically.
* Testing edge cases such as empty inputs, invalid `k` values, invalid overlap values, and invalid hybrid weights helps prevent failures.
* Fake embedders and fake chunkers make the tests faster and more isolated because they avoid making unnecessary API calls or model loading.
* Testing the saved `records.jsonl` and `embeddings.npy` files confirms that the index is being built correctly.
* CI makes the repository more production-ready because every push and pull request is automatically checked.

## Milestone 11 — FastAPI application

Implemented:

* Added a FastAPI API around the RAG system
* Created a `/query` endpoint for asking questions
* Added Pydantic request and response models
* Allowed the API user to choose:
  * query
  * retriever type
  * number of retrieved chunks
* Returned:
  * generated answer
  * retrieved source metadata
  * retrieval scores
* Tested the API using FastAPI's Swagger UI

Key lesson:

* Wrapping the project in an API makes it usable as a service rather than only as a local script.
* Pydantic models make the request and response structure explicit.
* Returning retrieved sources alongside the answer makes the system easier to inspect and debug.

## Milestone 12 — Dockerisation and deployment preparation

Implemented:

* Added a Dockerfile for containerising the FastAPI application
* Configured the container to run the API with Uvicorn
* Tested running the API inside Docker
* Mounted the local index into the container at runtime
* Prepared the project for AWS deployment using:
  * ECR for storing the Docker image
  * ECS/Fargate for running the API container
  * S3 for storing the persistent RAG index

Key lesson:

* Docker makes the application easier to run consistently across machines.
* Mounting the index at runtime keeps the image smaller and avoids baking generated data into the container.
* Separating the application image from the index storage on s3 is closer to how a real production RAG system would be deployed.

## Milestone 13 — AWS deployment and infrastructure cleanup

Implemented:

* Built and pushed the Docker image to AWS ECR
* Created ECS/Fargate infrastructure for running the API
* Configured task execution and task permissions
* Connected the deployed container to the required environment variables
* Used S3 as the persistent storage backend for the RAG index
* Cleaned up running resources after testing to avoid unnecessary AWS costs

Key lesson:

* Deploying the system made the project feel much closer to a real ML/LLM engineering project.
* ECS services manage running tasks and keep containers available.
* Task roles and execution roles separate application permissions from container startup permissions.
* Cleaning up cloud resources is an important part of deployment work.

## Final project status

I originally intended for this project to be a simple, local RAG prototype to help me with revision and to learn the basics of RAG.
However, by this stage, the project has grown into a much more sophisticated system with:
* structure-aware chunking
* metadata-aware records
* vector search
* BM25 keyword search
* hybrid retrieval with weighted Reciprocal Rank Fusion
* persistent index storage
* retrieval evaluation
* automated tests
* GitHub Actions CI
* FastAPI serving
* Docker support
* AWS deployment preparation using ECR, ECS/Fargate, and S3

## Future improvements

Potential future improvements include:

* Add an agentic RAG layer for query planning and retrieval strategy selection. This could include:
  * Add query rewriting when retrieval results are weak
  * Add evidence checking before answer generation
  * Allow an LLM to use tools, i.e., chooseing a retrieveal strategy.

* Add a small frontend for interacting with the deployed API

* Add more evaluation questions and compare retrieval strategies more formally
