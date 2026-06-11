from pathlib import Path
import json
import logging
import numpy as np
from typing import Any

from embeddings import Embeddings
from chunking import Chunking

logger = logging.getLogger(__name__)


class IndexManager:

    def __init__(
            self,
            embedder: Embeddings,
            chunker: Chunking,
            chunk_size: int = 300,
            overlap: int = 75,
            ):
        
        self.embedder = embedder
        self.chunker = chunker
        self.chunk_size = chunk_size
        self.overlap = overlap

    def build_index_from_data(
            self,
            data_path: Path = Path("data"),
            index_path: Path = Path("storage/index"),
            ) -> None:
        
        records: list[dict[str, Any]] = []
        embeddings: list[np.ndarray] = []

        # For each file
        for file in sorted(data_path.glob("*.md")):
            logger.info("Indexing %s ...", file.name)

            # Get the content as a string
            text = file.read_text(encoding="utf-8")

            # Get the chunks from this file
            chunks = self.chunker.chunk_text(
                text=text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

            # for each chunk
            for chunk_index, chunk in enumerate(chunks):
                # Embed the chunk
                embedding = self.embedder.embed(chunk)

                # Create a record from this chunk
                record = {
                    "chunk": chunk,
                    "metadata": {
                        "source_file": file.name,
                        "chunk_index": chunk_index,
                    },
                }

                # Append the record and embedding.
                records.append(record)
                embeddings.append(embedding)

        if not records:
            raise ValueError(f"No markdown chunks were found in {data_path}")

        # Once these records and embeddings are populated, save this state.
        self._save_index(
            records=records,
            embeddings=embeddings,
            index_path=index_path,
        )

    def _save_index(
            self,
            records: list[dict[str, Any]],
            embeddings: list[np.ndarray],
            index_path: Path,
            ) -> None:
        
        # Makes the directory represented by dir_path. 
        # parents=True means if any parent directories are missing, create those too.
        # So if neither folder exists: storage/index then Python creates storage/ then create storage/index/
        # exist_ok=True means if this directory already exists, then fine, don't crash.
        index_path.mkdir(parents=True, exist_ok=True)

        # Creates path objects to the records and embeddings files. 
        # Note, the / operator is special for Path objects. It joins paths and creates another path object.
        records_path = index_path / "records.jsonl"
        embeddings_path = index_path / "embeddings.npy"

        # Adds each chunk to the records.jsonl file. write mode overwrites the file if it is already made.
        with records_path.open("w", encoding="utf-8") as file:
            for record in records:
                json_line = json.dumps(record)
                file.write(json_line + "\n")

        # Save the embeddings as an embedding matrix.
        # turns our embedding list into a 2d matrix
        if embeddings:
            embeddings_matrix = np.vstack(embeddings)
        else:
            # If we have no embeddings, just create a empty np array.
            embeddings_matrix = np.array([])

        # Save out embedding matrix to embeddings_path
        np.save(embeddings_path, embeddings_matrix)

        logger.info(
            "Index saved | records=%d | records_path=%s | embeddings_path=%s",
            len(records),
            records_path,
            embeddings_path,
        )