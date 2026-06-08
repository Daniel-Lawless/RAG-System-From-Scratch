from pathlib import Path
import json
import logging
import numpy as np

from vector_search import VectorSearch
from embeddings import Embeddings
from chunking import Chunking

logger = logging.getLogger(__name__)

class IndexManager:

    def __init__(self,
                vector_search:VectorSearch,
                embedder : Embeddings,
                chunker : Chunking
                ):
        
        self.vector_search = vector_search
        self.embedder = embedder
        self.chunker = chunker

    # Populates the vector_search
    def index_text(self, text: str, source_file: str) -> None:

        # Split data into chunks
        chunks = self.chunker.chunk_text(text, chunk_size = 300, overlap=75)

        # Assign each chunk a chunk_id
        for chunk_index, chunk in enumerate(chunks):
            chunk_embedding = self.embedder.embed(chunk)
            self.vector_search.add_record(chunk = chunk,
                                      embedding = chunk_embedding,
                                      source_file = source_file,
                                      chunk_index = chunk_index
                                      )

    # Adds persistant storage
    def save(self, path: Path) -> None:

        # Makes the directory represented by dir_path. 
        # parents=True means if any parent directories are missing, create those too.
        # So if neither folder exists: storage/index then Python creates storage/ then create storage/index/
        # exist_ok=True means if this directory already exists, then fine, don't crash.
        path.mkdir(parents=True, exist_ok=True)

        # Creates path objects to the records and embeddings files. 
        # Note, the / operator is special for Path objects. It joins paths and creates another path object.
        records_path = path / "records.jsonl"
        embeddings_path = path / "embeddings.npy"

        # Adds each chunk to the records.jsonl file. write mode overwrites the file if it is already made.
        with records_path.open("w", encoding="utf-8") as file:
            for record in self.vector_search.records:
                json_line = json.dumps(record) # Converts a Python dict into a json object.
                file.write(json_line + "\n")
        
        # Save the embeddings as an embedding matrix.
        # turns our embedding list into a 2d matrix
        self.vector_search._rebuild_embedding_matrix()

        # If we do not have an embedding matrix, save an empty array.
        if self.vector_search.embeddings_matrix is None:
            np.save(embeddings_path, np.array([]))
        # else save the embedding matrix.
        else:
            np.save(embeddings_path, self.vector_search.embeddings_matrix)

        logger.info("Records saved to %s | Embeddings saved to %s",
                    records_path,
                    embeddings_path
                    )
    
    # Load state
    def load(self, path: Path) -> None:

        # File paths
        records_path = path / "records.jsonl"
        embeddings_path = path / "embeddings.npy"

        # Reset database
        self.vector_search.records = []
        self.vector_search.embeddings = []
        self.vector_search.embeddings_matrix = None

        logger.info("Loading records and embeddings...")

        # Convert each json "dict" in records.jsonl to a Python dict and add it to our records.  
        with records_path.open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                self.vector_search.records.append(record)
        
        # Load matrix from embeddings.npy
        loaded_matrix = np.load(embeddings_path)

        # If the file is empty
        if (loaded_matrix.size == 0):
            self.vector_search.embeddings_matrix = None
            self.vector_search.embeddings = []
        # else set matrix equal to the loaded matrix and populate embeddings.
        else:
            self.vector_search.embeddings_matrix = loaded_matrix
            self.vector_search.embeddings = [row for row in self.vector_search.embeddings_matrix]

        # Matrix is already loaded, so no need to rebuild
        self.vector_search._matrix_needs_rebuild = False

        logger.info("Records and embeddings loaded.")
