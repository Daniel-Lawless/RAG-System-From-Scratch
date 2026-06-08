from vector_search import VectorSearch
from chunking import Chunking
from embeddings import Embeddings
from openai import OpenAI
from typing import Any
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class RAGPipeline:

    # Initialise our vector db, our chunker, and our embedder.
    def __init__(self):
        self.client = OpenAI()

    # Build the context for the model
    def _build_context(self, results: list[dict[str, Any]]) -> str:
        retrieved_texts = []

        for record in results:
            chunk = record["chunk"]
            score = record["score"]
            retriever = record["retriever"]
            metadata = record["metadata"]

            retrieved_texts.append(
                f"[Score: {score:.4f}, retriever: {retriever}, "
                f"source_file: {metadata['source_file']}, chunk_index: {metadata['chunk_index']}]\n"
                f"{chunk}"
            )

        return "\n\n---\n\n".join(retrieved_texts)

    # Models response
    def response(self, query: str, results: list[dict[str,Any]]) -> str:

        if not results:
            logger.warning("No results were passed to the RAG pipeline.")
            return "I do not know."

        # Take the results from the retrieval method and build the context from them.
        context = self._build_context(results=results)

        # For debugging
        logger.debug("Context for model: %s", context)

        # Construct user query
        user_query = f"""
        Use the following retrieved context to answer the question. 

        question:
        {query}

        context:
        {context}

        Answer:
        """

        # Generate the response from the LLM
        response = self.client.responses.create(
            model = "gpt-5.4-nano",
            instructions = (
                    "You're a helpful assistant. Answer only using the provided context. "
                    "If the answer is not in the context, say you do not know."
            ),
            input=user_query
        )

        # Return the response from the LLM
        return response.output_text
