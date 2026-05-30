import numpy as np
from sentence_transformers import SentenceTransformer

class Embeddings:

    def __init__(self):
        
        # Model to embed strings.
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, text: str) -> np.ndarray:

        # Embed the vector
        vector_embedding = self.model.encode(text, convert_to_numpy=True)

        # Normalise the vector embedding so is it's just the dot product for cosine similarity search.
        vector_embed_normalized = vector_embedding / np.linalg.norm(vector_embedding)

        # Return the normalized embeded vector.
        return vector_embed_normalized