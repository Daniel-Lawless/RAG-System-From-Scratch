import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
load_dotenv()
REQUIRED_INDEX_FILES = ("records.jsonl", "embeddings.npy")


def _validate_index_path(index_path : Path) -> None:
    # If a file does not exist, add it to the missing_files list.
    missing_files = [
        filename
        for filename in REQUIRED_INDEX_FILES
        if not (index_path / filename).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"missing files in {index_path}: {missing_files}"
        )
    
# An s3_key is the path to a file in an s3 bucket.
# So for s3://bucket_name/storage/index/record.json,
# the key is storage/index/record.json
def _s3_key(prefix: str, filename: str) -> str:
    # Removes the first and last slash to prevent double slashes
    prefix = prefix.strip("/")

    if prefix:
        return f"{prefix}/{filename}"
    
    return filename

def _download_index_from_s3(bucket:str, prefix:str, index_path: Path) -> None:
    # This creates the local folder where the downloaded index files will be stored
    index_path.mkdir(parents=True, exist_ok=True)

    # Connects to s3. Allows Python to talk to S3.
    s3_client = boto3.client("s3")

    for filename in REQUIRED_INDEX_FILES:
        key = _s3_key(prefix, filename)
        destination_path = index_path / filename

        try:
            # Tells s3 which file to download. destination is a Path object, 
            # but boto3 expects a string path, so we convert it to a string.
            s3_client.download_file(bucket, key, str(destination_path))
            logger.info(f"Downloaded s3://{bucket}/{key} to {destination_path}")

        except ClientError as error:
            raise RuntimeError(
                f"Failed to download s3://{bucket}/{key}"
            ) from error # <- this keeps the original boto3 error as well.

# This tells us where to get the index from.
def prepare_index() -> Path:
    # Looks in .env for our ENV INDEX_PATH variable, it if does not exist it defaults to "local"
    index_backend = os.getenv("INDEX_BACKEND", "local").lower()

    logger.info("Preparing index from: %s...", index_backend)

    if index_backend == "local":
        index_path = Path(os.getenv("INDEX_PATH", "storage/index"))
        _validate_index_path(index_path)
        return index_path
    
    if index_backend == "s3":
        # Set up for download to s3
        bucket = os.getenv("S3_BUCKET")
        prefix = os.getenv("S3_PREFIX", "rag-index")
        index_path = Path(os.getenv("INDEX_PATH", "/tmp/index"))

        if not bucket:
            raise ValueError("S3_BUCKET must be set when INDEX_BACKEND=s3")
        
        # Download the saved index on s3
        _download_index_from_s3(
            bucket=bucket,
            prefix=prefix,
            index_path=index_path
        )

        _validate_index_path(index_path)
        return index_path

    raise ValueError(
        f"Unsupported INDEX_BACKEND: {index_backend}. "
        "Use 'local' or 's3'"
        )
