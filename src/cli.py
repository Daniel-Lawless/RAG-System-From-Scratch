import argparse
import logging
from pathlib import Path
import shutil

from vector_search import VectorSearch
from chunking import Chunking
from embeddings import Embeddings
from rag_pipeline import RAGPipeline
from keyword_search import KeywordSearch
from index_manager import IndexManager

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class RAGCLI:

    def __init__(self, index_path: Path = Path("storage/index"), data_path: Path = Path("data")):
        self.index_path = Path(index_path)
        self.data_path = Path(data_path)

    def _create_index_manager(self) -> tuple[VectorSearch, Embeddings, IndexManager]:
        vector_search = VectorSearch()
        embedder = Embeddings()
        chunker = Chunking()

        index_manager = IndexManager(vector_search=vector_search,
                                     embedder=embedder,
                                     chunker=chunker)

        return (vector_search, embedder, index_manager)
    
    def _create_rag_pipeline(self) -> RAGPipeline:
        return RAGPipeline()
    
    # Checks whether we have a saved state.
    def _index_exists(self) -> bool:
        records_path = self.index_path / "records.jsonl"
        embeddings_path = self.index_path / "embeddings.npy"

        # Returns true if both files exist and they have more than 0 bytes, so we have a saved state.
        return ((records_path.exists() and records_path.stat().st_size > 0) and
                (embeddings_path.exists() and embeddings_path.stat().st_size > 0)
                )
    
    # Command to build state
    def index(self, force: bool = False) -> None: 
        
        # If a state already exists and you're not forcing a rebuild, tell the user.
        if self._index_exists() and not force:
            logger.warning(
                "Index already exists at %s. Use --force to rebuild.",
                self.index_path
            )
            return
        
        # If we want to force a rebuild, delete the index directory and rebuild it.
        if self._index_exists() and force:
            # from "storage/index" this removes the index directory and everything below it, so record.jsonl and embeddings.npy
            shutil.rmtree(self.index_path)
            logger.info("State removed.")
        
        _, _, index_manager = self._create_index_manager()

        logger.info("No previous state found. Creating new state:")


        for file in sorted(self.data_path.glob("*.md")):

            logger.info(
                "indexing %s ...",
                file.name
            )

            text = file.read_text(encoding="utf-8")

            index_manager.index_text(
                text=text,
                source_file=file.name
            )
            
        # Save this state for next time.
        index_manager.save(self.index_path)
    
    # Command to clear state.
    def clear(self):
        # If a state does not exist, we cannot delete it.
        if not self._index_exists():
            logger.warning(
                "No state exists at %s",
                self.index_path
            )
            return

        # If the state exists, remove it.
        shutil.rmtree(self.index_path)
        logger.info("Removed state at %s", self.index_path)

    # Command to ask a query
    def ask(self, query: str, num_retrieved_chunks: int = 10, retrieval_method: str = "vector") -> None:

        # Is not state exists, ask user to build one.
        if not self._index_exists():
            logger.warning("No state exists. Run: python3 src/cli.py index")
            return
        
        if retrieval_method == "vector":

            logger.info("Starting vector search...")

            vector_search, embedder, index_manager = self._create_index_manager()
            rag_pipeline = self._create_rag_pipeline()

            # Load saved state.
            index_manager.load(self.index_path)

            # Embed the query
            query_embedding = embedder.embed(text=query)

            # Return the k most similar records
            results = vector_search.search(
                query_embedding=query_embedding,
                k=num_retrieved_chunks
            )

            logger.info("Vector search complete")

            # Pass those records into response to build context and retrieve the LLM response.
            answer = rag_pipeline.response(
                query=query,
                results=results
            )

            # Print model response to the terminal
            print(f"Model response: \n{answer}")
        
        elif retrieval_method == "keyword":

            logger.info("Starting keyword search...")

            rag_pipeline = self._create_rag_pipeline()

            # Populate keyword search with saved state.
            keyword_search = KeywordSearch.from_jsonl(self.index_path) 

            # Get the k records with the highest score.
            results = keyword_search.search(
                query=query,
                k=num_retrieved_chunks
            )

            logger.info("Keyword search complete...")

            # Pass those records into response to build context and retrieve the LLM response.
            answer = rag_pipeline.response(
                query=query,
                results=results,
            )

            # Print model response to the terminal
            print(f"Model response: \n{answer}")

        elif retrieval_method == "hybrid":
            raise NotImplementedError("Hybrid not implemented yet.")

# Build our CLI parser.
def build_parser() -> argparse.ArgumentParser:

    logger.info("Attempting to build parser...")

    # This creates the main CLI program. 
    parser = argparse.ArgumentParser(description="Mini RAG System CLI")

    # This means this CLI will have subcommands, i.e., index, ask, clear. 
    # Required means the user must provide a command. So python src/cli.py is invalid.
    # dest means when we write a command in the terminal, it is saved to args.command
    subparser = parser.add_subparsers(
        dest="command",
        required=True
    )

    # This allows us to write commands like python src/cli.py ask
    index_parser = subparser.add_parser("index")
    ask_parser = subparser.add_parser("ask")
    clear_parser = subparser.add_parser("clear")

    # Add arguments for the ask sub parser.
    ask_parser.add_argument(
        "query",
        type=str,
        help="Ask a question"
    )
    
    ask_parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Enter the number of chunks you want to retrieve"
    )
    
    ask_parser.add_argument(
        "--retriever",
        type=str,
        choices=["vector", "keyword", "hybrid"],
        default="vector",
        help="choose retrieval method"
    )

    # Add argument for the index parser. action = "store_true" means if we do not include --force, i.e., python src/cli.py index then args.force == False
    # If we do include it, python src/cli.py index --force, then args.force == True.
    index_parser.add_argument(
        "--force",
        action="store_true",
        help="True to force rebuilding the vector database, False otherwise"
    )

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
        cli.ask(query=args.query,
                num_retrieved_chunks=args.k,
                retrieval_method=args.retriever
                )
    
if __name__ == "__main__":
    main()

