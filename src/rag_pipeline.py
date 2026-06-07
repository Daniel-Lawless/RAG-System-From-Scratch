from vector_db import VectorDB
from chunking import Chunking
from embeddings import Embeddings
from openai import OpenAI
from pathlib import Path
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class RAGPipeline():

    # Initialise our vector db, our chunker, and our embedder.
    def __init__(self, vector_db: VectorDB, chunker: Chunking, embedder: Embeddings):
        self.vector_db = vector_db
        self.chunker = chunker
        self.embedder = embedder
    
    # Populates the vector database.
    def index_text(self, text: str, source_file: str) -> None:

        # Split data into chunks
        chunks = self.chunker.chunk_text(text, chunk_size = 300, overlap=75)

        # Assign each chunk a chunk_id
        for chunk_index, chunk in enumerate(chunks):
            chunk_embedding = self.embedder.embed(chunk)
            self.vector_db.add_record(chunk = chunk,
                                      embedding = chunk_embedding,
                                      source_file = source_file,
                                      chunk_index = chunk_index
                                      )

    # Models response
    def response(self, query: str, num_retrieve: int) -> str:

        # For debugging
        logger.debug("User passed query: %s", query)

        # Embed the query
        query_embedding = self.embedder.embed(query)

        # Retrieve num_retrieve most similar chunks. 
        most_similar = self.vector_db.search(query_embedding, num_retrieve)

        # For debugging
        for similarity, record in most_similar:
            logger.debug(
                # f-string: build message first, then maybe discard it
                # logging placeholders: check log level first, only build message if needed
                "similarity=%.4f | Retrieved chunk=%s | source_file=%s | chunk_index=%s",
                similarity,
                record["chunk"][:10],
                record["metadata"]["source_file"],
                record["metadata"]["chunk_index"]
            )

        retrieved_texts = []

        # Return the source_fle, chunk_index, and chunk.
        for _, record in most_similar:
            source_file = record["metadata"]["source_file"]
            chunk_index = record["metadata"]["chunk_index"]
            chunk = record["chunk"]

            retrieved_texts.append(
                f"[Source file: {source_file}, chunk index: {chunk_index}]\n"
                f"{chunk}"
            )


        # Model context is a combination of the chunk text and important meta data which can be used by the model to reference where the information came from
        context = "\n\n --- \n\n".join(retrieved_texts)

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
        client = OpenAI()
        response = client.responses.create(
            model = "gpt-5.4-nano",
            instructions = (
                    "You're a helpful assistant. Answer only using the provided context. "
                    "If the answer is not in the context, say you do not know."
            ),
            input=user_query
        )

        # Return the response from the LLM
        return response.output_text
    
if __name__ == "__main__":

    # initialise vector database, chunker, and embedder
    vector_db = VectorDB()
    chunker = Chunking()
    embedder = Embeddings()
    
    # pass them to our RAG pipeline
    rag_pipeline = RAGPipeline(vector_db, chunker, embedder)

    # Specify the path to the directory that contains the data we want to store in our vector DB.
    data_dir = Path("data")

    # Specify the path to the files that contain saved chunks and embeddings.
    index_dir = Path("storage/index")
    chunks_path = index_dir / "chunks.jsonl"
    embeddings_path = index_dir / "embeddings.npy"

    # If a previous saved state exists, load that state.
    if (chunks_path.exists() 
        and embeddings_path.exists()
        # checks if the files are non-empty, i.e., if it has more than 0 bytes.
        and chunks_path.stat().st_size > 0
        and embeddings_path.stat().st_size > 0):

        logger.info("Saved state exists. Loading previous state. ")
        vector_db.load(index_dir) 

    # Else create that state and save it for next time.
    else:
        logger.info("No previous state found. Creating new state:")
        # For each file in this directory, read its contents and index it. This populates our database.
        for file in sorted(data_dir.glob("*.md")):
            logger.info(f"Indexing {file.name}...")
            text = file.read_text(encoding = "utf-8") # Returns the entire file as a Python string
            rag_pipeline.index_text(text, source_file = file.name)

        # Save this state.
        vector_db.save(index_dir)
        logger.info("New state created.")

    # Once the DB is populated, return the models response.
    answer = rag_pipeline.response("What is the German Tank Problem?", 10)
    print(answer)
