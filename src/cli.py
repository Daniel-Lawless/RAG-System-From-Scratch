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
from hybrid_search import HybridSearch

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class RAGCLI:

    def __init__(self, index_path: Path = Path("storage/index"), data_path: Path = Path("data")):
        self.index_path = Path(index_path)
        self.data_path = Path(data_path)

    def _create_index_manager(self) -> IndexManager:
        embedder = Embeddings()
        chunker = Chunking()

        return IndexManager(
            embedder=embedder,
            chunker=chunker,
    )
    
    def _create_rag_pipeline(self) -> RAGPipeline:
        return RAGPipeline()
    
    # Checks whether we have a saved index.
    def _index_exists(self) -> bool:
        records_path = self.index_path / "records.jsonl"
        embeddings_path = self.index_path / "embeddings.npy"

        # Helper to check if a file exists and is not empty
        def is_valid(path):
            return path.is_file() and path.stat().st_size > 0

        return is_valid(records_path) and is_valid(embeddings_path)
    
    # If no index exists, attempt to build it.
    def _ensure_index_exists(self) -> bool:
        if self._index_exists():
            return True

        logger.info("No index found. Building index first...")
        self.index(force=False)

        # Check the build succeeded
        if not self._index_exists():
            logger.warning(
                "Index could not be created. Check that %s contains markdown files.",
                self.data_path,
            )
            return False

        return True
    
    # Command to build index
    def index(self, force: bool = False) -> None: 
        # If a user attempts to to build an index when one already exists,
        # tell them how to force rebuild.
        if self._index_exists():
            if not force:
                logger.warning(
                    "Index already exists at %s. Use --force to rebuild.",
                    self.index_path
                )
                return
            
            # If we reach here, an index exists and force is True,
            # so delete the current saved index.
            shutil.rmtree(self.index_path)
            logger.info("Previous index removed.")
        
        # Build the new index
        logger.info("Creating new index...")
        index_manager = self._create_index_manager()

        index_manager.build_index_from_data(
            data_path=self.data_path, 
            index_path=self.index_path
        )

        logger.info("New index created.")
                    
    # Command to clear index.
    def clear(self):
        # If a index does not exist, we cannot delete it.
        if not self.index_path.exists():
            logger.warning(
                "No index exists at %s",
                self.index_path
            )
            return

        # If the index exists, remove it.
        shutil.rmtree(self.index_path)
        logger.info("Removed index at %s", self.index_path)

    # Command to ask a query
    def ask(
        self,
        query: str,
        num_retrieved_chunks: int = 10,
        retrieval_method: str = "vector",
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0
    ) -> None:

        # If no index exists, build the index.
        if not self._ensure_index_exists():
            return
        
        # Set up the pipeline and a placeholder for results
        rag_pipeline = self._create_rag_pipeline()
        results = []
        
        # Fetch the results using the requested retriever
        if retrieval_method == "vector":

            # Populate vector search.
            vector_search = VectorSearch.from_index(self.index_path)

            # Create embedder
            embedder = Embeddings()

            # Embed the query
            query_embedding = embedder.embed(text=query)
            
            # Results from vector search
            results = vector_search.search(
                query_embedding=query_embedding,
                k=num_retrieved_chunks
            )
            
        elif retrieval_method == "keyword":

            # Populate keyword search.
            keyword_search = KeywordSearch.from_jsonl(self.index_path) 
            
            # Results from keyword search
            results = keyword_search.search(
                query=query,
                k=num_retrieved_chunks
            )

        elif retrieval_method == "hybrid":

            # Populate vector and keyword search.
            vector_search = VectorSearch.from_index(self.index_path)
            keyword_search = KeywordSearch.from_jsonl(self.index_path)

            # Create embedder
            embedder = Embeddings()
            
            hybrid_search = HybridSearch(
                vector_search=vector_search,
                keyword_search=keyword_search,
                embedder=embedder,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight
            )
            
            # Results from hybrid search
            results = hybrid_search.search(
                query=query,
                k=num_retrieved_chunks
            )

        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")

        # Pass the fetched records to the LLM and print the models response
        answer = rag_pipeline.response(
            query=query,
            results=results
        )

        print(f"Model response: \n{answer}")

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

    ask_parser.add_argument(
        "--vector_weight",
        type=float,
        default=1,
        help="Weight for vector search in hybrid retrieval"
    )

    ask_parser.add_argument(
        "--keyword_weight",
        type=float,
        default=1,
        help="Weight for keyword search in hybrid retrieval"
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
                retrieval_method=args.retriever,
                vector_weight=args.vector_weight,
                keyword_weight=args.keyword_weight
                )
    
if __name__ == "__main__":
    main()
