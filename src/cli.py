import argparse
import logging
import shutil
import boto3
import os

from dotenv import load_dotenv
from botocore.exceptions import ClientError
from pathlib import Path
from vector_search import VectorSearch
from chunking import Chunking
from embeddings import Embeddings
from rag_pipeline import RAGPipeline
from keyword_search import KeywordSearch
from index_manager import IndexManager
from hybrid_search import HybridSearch
from logging_config import configure_logging

load_dotenv()
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
    
    # An s3_key is the path to a file in an s3 bucket.
    # So for s3://bucket_name/storage/index/record.json,
    # the key is storage/index/record.json
    def _s3_key(self, prefix: str, filename: str) -> str:
        # Removes the first and last slash to prevent double slashes
        prefix = prefix.strip("/")

        if prefix:
            return f"{prefix}/{filename}"
        
        return filename
    
    def _s3_index_exists(self, s3_bucket:str, s3_prefix:str) -> bool:
        # Set up a connection to S3
        s3_client = boto3.client("s3")

        # We want to check if these files exist on s3
        required_files = ("embeddings.npy", "records.jsonl")

        for filename in required_files:
            # get the key
            key = self._s3_key(s3_prefix, filename)

            try:
                # This checks info, if info exists, the file exists.
                s3_client.head_object(Bucket=s3_bucket, Key=key)

            except ClientError:
                # If s3 object had no information,
                # the index does not exist correctly
                return False
        
        # If all files were there, return true
        return True
    
    def _upload_index_to_s3(self, bucket:str, prefix:str) -> None:
        s3_client = boto3.client("s3")

        required_files = ("embeddings.npy", "records.jsonl")

        for filename in required_files:
            # Where the file is on our machine
            local_path = self.index_path / filename
            # Where is needs to go inside the bucket
            key = self._s3_key(prefix, filename)

            if not local_path.is_file():
                raise FileNotFoundError(f"File {filename} does not exist at {local_path}")
            
            # This uploads a file from our machine to s3
            s3_client.upload_file(
                str(local_path),
                bucket,
                key
            )

            logger.info("Uploaded %s to s3://%s/%s", local_path, bucket, key)

    def _remove_s3_index(self, s3_bucket:str, s3_prefix:str) -> None:
        s3_client = boto3.client("s3")

        required_files = ("embeddings.npy", "records.jsonl")

        for filename in required_files:
            key = self._s3_key(s3_prefix, filename)

            try:
                s3_client.delete_object(
                    Bucket=s3_bucket,
                    Key=key
                )
                logger.info("Deleted s3://%s/%s", s3_bucket, key)

            except ClientError as error:
                raise RuntimeError(
                    f"Tried deleting index from s3://{s3_bucket}/{key}, but no such index exists"
                    ) from error

    # Command to build index
    def index(
        self,
        force: bool = False,
        backend: str = "local",
        s3_bucket: str | None = None,
        s3_prefix: str = "rag-index",
    ) -> None:
        
        if backend == "local":
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
            
            return
        
        elif backend == "s3":
            # Define bucket and prefix. It will get the cli value first, if it is empty and
            # a .env value is provided, it will then take that, else it returns None
            s3_bucket = s3_bucket or os.getenv("S3_BUCKET")
            s3_prefix = s3_prefix or os.getenv("S3_PREFIX", "rag-index")

            if not s3_bucket:
                raise ValueError(
                    f"S3 bucket must be provided either from --s3_bucket or S3_BUCKET in .env"
                )
            
            # Removes trailing spaces in the bucket name
            s3_bucket = s3_bucket.strip()
            # Removes the first and last slash to prevent double slashes
            s3_prefix = s3_prefix.strip("/")

            # If the index already exists
            if self._s3_index_exists(
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix
            ):
                # And force is false
                if force == False:
                    logger.warning("" \
                        "Index already exists at s3://%s/%s",
                        s3_bucket,
                        s3_prefix,
                    )

                    return

                logger.info(
                    "Rebuilding index at s3://%s/%s because force = %d",
                    s3_bucket,
                    s3_prefix,
                    force
                )
            
            # Remove the local version so we know it is fresh
            if self._index_exists():
                shutil.rmtree(self.index_path)
                logger.info("previous local index at %s was removed", self.index_path)


            logger.info("Creating new index locally before uploading to S3...")

            # Create index manager object
            index_manager = self._create_index_manager()

            # Build index from new and save it to self.index_path
            index_manager.build_index_from_data(
                data_path=self.data_path,
                index_path=self.index_path
            )

            # Upload newly created index to s3.
            logger.info("Uploading index to S3...")
            self._upload_index_to_s3(
                bucket=s3_bucket,
                prefix=s3_prefix
            )

            logger.info("New S3 index created at s3://%s/%s", s3_bucket, s3_prefix)
            return

        raise ValueError(
            f"unsupported backend value: {backend} \available options: 'local', 's3'"
            )

    # Command to clear index.
    def clear(
        self,
        backend:str,
        s3_bucket:str | None = None,
        s3_prefix:str | None = None
        ) -> None:

        # If the index is stored locally
        if backend == "local":
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
        
        # If the backend is stored on AWS s3
        elif backend == "s3":
            # Get bucket and prefix
            s3_bucket = os.getenv("S3_BUCKET")
            s3_prefix = os.getenv("S3_PREFIX", "rag-index")

            if not s3_bucket:
                raise ValueError(
                    "S3 bucket must be provided with --s3-bucket or S3_BUCKET in .env"
                    )
            
            s3_bucket = s3_bucket.strip()
            s3_prefix = s3_prefix.strip("/")

            if not self._s3_index_exists(
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix
            ):
                logger.warning(
                    "No index at s3://%s/%s exists",
                    s3_bucket,
                    s3_prefix
                )

            # Remove the index on s3
            self._remove_s3_index(
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix
            )

            logger.info("Removed S3 index at s3://%s/%s", s3_bucket, s3_prefix)
            return

        raise ValueError(
            f"unsupported backend value: {backend} \available options: 'local', 's3'"
            )


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

    # If --backend is s3, create and push the index to s3, if it is local, build it on our machine
    index_parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "s3"],
        default="local",
        help="Choose where to build the index."
    )

    index_parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket to upload the index to when using --backend s3.",
    )

    index_parser.add_argument(
        "--s3-prefix",
        type=str,
        default="rag-index",
        help="S3 prefix/folder to upload the index to when using --backend s3.",
    )

    clear_parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "s3"],
        default="local",
        help="Choose which index to remove"
    )
    
    clear_parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket to delete the index from when using --backend s3.",
    )

    clear_parser.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        help="S3 prefix/folder to delete the index from when using --backend s3.",
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
        cli.index(
            force=args.force,
            backend=args.backend,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix
            )
    
    # If the command was clear, run clear().
    elif args.command == "clear":
        cli.clear(
            backend = args.backend,
            s3_bucket= args.s3_bucket,
            s3_prefix=args.s3_prefix
            )
    
    # If the command was ask, pass the arguments into ask.
    elif args.command == "ask":
        cli.ask(
            query=args.query,
            num_retrieved_chunks=args.k,
            retrieval_method=args.retriever,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight
        )
    
if __name__ == "__main__":
    configure_logging()
    main()
