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

    def __init__(self, index_path: Path = Path("storage/index"), data_path: Path = Path("data")):
        self.index_path = Path(index_path)
        self.data_path = Path(data_path)

    # We should only build the pipeline when needed. For instance, for the clear command
    # we don't want to create the embedding model, so we do not call this function.
    def _create_pipeline(self) -> tuple[VectorDB, RAGPipeline]:
        vector_db = VectorDB()
        chunker = Chunking()
        embedder = Embeddings()

        rag_pipeline = RAGPipeline(
            vector_db,
            chunker,
            embedder
        )

        return (vector_db, rag_pipeline)
    
    # Checks whether we have a saved state.
    def _index_exists(self) -> bool:
        chunks_path = self.index_path / "chunks.jsonl"
        embeddings_path = self.index_path / "embeddings.npy"

        # Returns true if both files exist and they have more than 0 bytes, so we have a saved state.
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
            # from "storage/index" this removes the index directory and everything below it, so chunks.jsonl and embeddings.npy
            shutil.rmtree(self.index_path)
            logger.info("State removed.")
        
        vector_db, rag_pipeline = self._create_pipeline()

        # If no state exists, build it.
        if not self._index_exists():
            logger.info("No previous state found. Creating new state:")
            for file in sorted(self.data_path.glob("*.md")):
                logger.info("indexing %s ...", file.name)
                text = file.read_text(encoding="utf-8")
                rag_pipeline.index_text(text=text, source_file=file.name)
            
            # Save this state for next time.
            vector_db.save(self.index_path)
    
    # Clear state.
    def clear(self):
        # If a state does not exist, we cannot delete it.
        if not self._index_exists():
            logger.warning("No state exists at %s", self.index_path)
            return

        # If the state exists, remove it.
        shutil.rmtree(self.index_path)
        logger.info("Removed state at %s", self.index_path)

    def ask(self, query: str, num_retrieved_chunks: int = 10) -> None:

        # Is not state exists, ask user to build one.
        if not self._index_exists():
            logger.warning("No state exists. Run: python3 src/cli.py index")
            return
        
        vector_db, rag_pipeline = self._create_pipeline()

        # Load saved state.
        vector_db.load(self.index_path)

        # Return and print the answer.
        answer = rag_pipeline.response(query, num_retrieved_chunks)
        print(answer)

# Build our CLI parser.
def build_parser() -> argparse.ArgumentParser:

    logger.info("Attempting to build parser...")

    # This creates the main CLI program. 
    parser = argparse.ArgumentParser(description="Mini RAG System CLI")

    # This means this CLI will have subcommands, i.e., index, ask, clear. 
    # Required means the user must provide a command. So python src/cli.py is invalid.
    # dest means when we write a command in the terminal, it is saved to args.command
    subparser = parser.add_subparsers(dest="command", required=True)

    # This allows us to write commands like python src/cli.py ask
    index_parser = subparser.add_parser("index")
    ask_parser = subparser.add_parser("ask")
    clear_parser = subparser.add_parser("clear")

    # Add arguments for the ask sub parser.
    ask_parser.add_argument("query", type=str, help="Ask a question")
    ask_parser.add_argument("-k", type=int, default=10, help="Enter the number of chunks you want to retrieve")

    # Add argument for the index parser. action = "store_true" means if we do not include --force, i.e., python src/cli.py index then args.force == False
    # If we do include it, python src/cli.py index --force, then args.force == True.
    index_parser.add_argument("--force", action="store_true", help="True to force rebuilding the vector database, False otherwise")

    logger.info("Parser built.")

    return parser

def main() -> None:
    
    # Build the parser and extract arguments the user used.
    parser = build_parser()
    args = parser.parse_args()

    # Create CLI object
    cli = RAGCLI()
    
    # If the command was index, pass the force argument into index.
    if args.command == "index":
        cli.index(force=args.force)
    
    # If the command was clear, run clear().
    elif args.command == "clear":
        cli.clear()
    
    # If the command was ask, pass the arguments into ask.
    elif args.command == "ask":
        cli.ask(query=args.query, num_retrieved_chunks=args.k)
    
if __name__ == "__main__":
    main()
