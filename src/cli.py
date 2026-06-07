import argparse
import logging
from pathlib import Path
import shutil

from vector_db import VectorDB
from chunking import Chunking
from embeddings import Embeddings
from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class RAGCLI():

    def __init__(self,
                vector_db : VectorDB,
                chunker: Chunking,
                embedder: Embeddings,
                index_path: Path = Path("storage/index"),
                data_path: Path = Path("data")):
        
        self.vector_db = vector_db
        self.chunker = chunker
        self.embedder = embedder
        self.index_path = index_path
        self.data_path = data_path
        self.rag_pipeline = RAGPipeline(vector_db, chunker, embedder)
    
    # Checks whether we have a saved state.
    def _index_exists(self) -> bool:
        chunks_path = self.index_path / "chunks.jsonl"
        embeddings_path = self.index_path / "embeddings.npy"

        # returns true if we both files exist and they have more than 0 bytes, so we have a saved state.
        return ((chunks_path.exists() and chunks_path.stat().st_size > 0) and
                (embeddings_path.exists() and embeddings_path.stat().st_size > 0)
                )
    
    # Build vector database
    def index(self, force: bool = False) -> None: 
        
        # If a state already exists and you're not forcing a rebuild, tell the user.
        if self._index_exists() and not force:
            logger.warning("Index already exists at %s. Use --force to rebuild.", self.index_path)
            return
        
        # If we want to force a rebuild, delete the index directory and rebuild it.
        if self._index_exists() and force:
            # from "storage/index" this removes the index directory and everything below it, so chunks.jsonl and embedding.npy
            shutil.rmtree(self.index_path)
            logger.info("State removed.")

        # If no state exists, build it.
        if not self._index_exists():
            logger.info("No previous state found. Creating new state:")
            for file in sorted(self.data_path.glob("*.md")):
                logger.info("indexing %s ...", file.name)
                text = file.read_text(encoding="utf-8")
                self.rag_pipeline.index_text(text=text, source_file=file.name)
            
            # Save this state for next time.
            self.vector_db.save(self.index_path)
    
    # Clear state.
    def clear(self):
        # If a state does not exists, we cannot delete it.
        if not self._index_exists():
            logger.warning("No state exists at %s", self.index_path)
            return

        # If the state exists, remove it.
        shutil.rmtree(self.index_path)
        logger.info("Removed state at %s", self.index_path)

    def ask(self, query: str, num_retrived_chunks: int = 10) -> None:

        # Is not state exists, ask user to build one.
        if not self._index_exists():
            logger.warning("No state exists. Run: python3 src/cli.py index")
            return

        # Load saved state.
        self.vector_db.load(self.index_path)

        # Return and print the answer.
        answer = self.rag_pipeline.response(query, num_retrived_chunks)
        print(answer)

            
