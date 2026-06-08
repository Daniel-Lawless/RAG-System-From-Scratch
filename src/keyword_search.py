import math
import re
from collections import Counter
from typing import Any
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class KeywordSearch:

    def __init__(self, k1: float = 1.5, b: float = 0.75):

        self.k1 = k1
        self.b = b

        # List of chunks
        self.chunks : list[str] = []
        # list of metadata. Dictionaries have string keys but any type of value
        self.metadata : list[dict[str, Any]] = [] 

        # Each chunk gets its own frequency dictionary. Helps answer "How strongly does this chunk talk about the query term"
        self.term_frequencies : list[Counter[str]] = []
        # Counts how many chunks contain that word. Helps answer "Is this query term rare or common across the whole corpus"
        self.document_frequencies : Counter[str] = Counter()
        # length of each chunk
        self.document_lengths: list[int] = []

        self.average_document_length = 0.0
        self.number_of_documents = 0

    @classmethod # The reason this is a class method is because it returns a new KeywordSearch object, but populated.
    def from_jsonl(cls, index_path: Path = Path("storage/index")) -> "KeywordSearch":
        keyword_search = cls() # cls() is convention, and it is equivalent to keyword_search = KeywordSearch()

        # Create path to persistent record records.jsonl
        records_path = index_path / "records.jsonl"

        # If the saved record state does not exist, return
        if not records_path.exists() or records_path.stat().st_size == 0:
            logger.warning("Path %s does not exist or is empty", records_path)
            raise FileNotFoundError(f"Path {records_path} does not exist or is empty")

        # Populate keyword search with our saved state.
        with records_path.open("r", encoding="utf-8") as file:
             for line in file:
                 record = json.loads(line)
                 keyword_search._add_chunk(record["chunk"], record["metadata"])

        logger.info(
            "Loaded %d chunks into keyword search from %s",
            keyword_search.number_of_documents,
            records_path,
        )

        return keyword_search 

    def _tokenise(self, text:str) -> list[str]:
        text = text.lower()

        # This will match if the user enters λ or lambda
        replacements = {
            "λ" : "λ lambda",
            "σ" : "σ sigma ",
            "μ": " μ mu ",
            "π": " π pi ",
        }

        # Replace any key in text with its replacement.
        for key, value in replacements.items():
            text = text.replace(key, value)
        
        # This splits our text into a list of words without punctuation.
        return re.findall(r"\b\w+\b", text)
    
    # This function is what populates our search
    def _add_chunk(self, chunk: str, metadata: dict[str, Any]) -> None:

        # Splits the chunk into list of its words.
        tokens = self._tokenise(chunk)

        logger.debug("Chunk tokens : %s", tokens)

        # If tokens is empty, then just return.
        if not tokens:
            return
        
        # Counts how many times each word in the chunk appears
        term_frequency = Counter(tokens)

        logger.debug("How often each term appears in this chunk: %s", term_frequency)

        self.chunks.append(chunk)
        self.metadata.append(metadata)
        self.term_frequencies.append(term_frequency)
        self.document_lengths.append(len(tokens))

        # Track how many chunks contain each unique term.
        for term in term_frequency:
            self.document_frequencies[term] += 1

        # Another chunk has been seen, so increment.
        self.number_of_documents += 1

        # Average document length.
        self.average_document_length = sum(self.document_lengths) / self.number_of_documents

        logger.debug(
        "Indexed chunk %d | length=%d | unique_terms=%d | metadata=%s | preview=%r",
        self.number_of_documents - 1,
        len(tokens),
        len(term_frequency),
        metadata,
        self._preview(chunk)
        )

        logger.info(
            "Keyword index now contains %d chunks | avg_length=%.2f",
            self.number_of_documents,
            self.average_document_length,
        )

    # Make debugging statements look nicer.
    def _preview(self, text: str, length: int = 80) -> str:
        return text[:length].replace("\n", " ") + ("..." if len(text) > length else "")

    # How rare is this chunk across all chunks. Rare terms are usually more useful for search.
    def _idf(self, term: str) -> float:

        # Extract how many chunks this term appears in.
        document_frequency = self.document_frequencies[term]

        # Counts how many chunks the term does not appear it. 0.5 terms are used to smoothen out edge cases.
        numerator = self.number_of_documents - document_frequency + 0.5

        # Give less weight to common terms.
        denominator = document_frequency + 0.5

        # Rare terms will have a high IDF, very common terms will have a low IDF. +1 is used so idf is non-negative.
        idf = math.log(1 + numerator / denominator)

        # For debugging
        logger.debug(
            "IDF term=%r | document_frequency=%d | Num_documents=%d | idf=%.4f",
            term,
            document_frequency,
            self.number_of_documents,
            idf,
        )

        return idf
    
    def _score_chunk(self, query_terms: list[str], chunk_index: int) -> float:
        # Initialise score.
        score = 0.0

        # How many times each word in this chunk occurs.
        term_frequency = self.term_frequencies[chunk_index] 

        # The length of this chunk
        document_length = self.document_lengths[chunk_index]

        for term in query_terms:

            # How often does this query term appear in this chunk.
            query_term_frequency = term_frequency[term]

            # If this query term is not in this chunk, go to the next term.
            if query_term_frequency == 0:
                continue

            # Calculate the idf of this term. I.e., how rare is this term in our corpus.
            # First half of the BM25 formula
            idf = self._idf(term)

            # Second half of the BM25 formula. 

            # k1 controls how much repeated terms matter.
            # self.k1 + 1, the +1 is used so that given the query_term_frequency in an 
            # average length chunk is 1, the score is approx idf * 1
            numerator = query_term_frequency * (self.k1 + 1)

            # self.b controls how much document/chunk length matters. The denominator controls
            # term frequency saturation, so a word occuring twice is better than once, but a word
            # appearing 50 times should not make the score explode, So the denominator makes the
            # gain from extra repetitions slow down. It also controls chunk length normalisation. 
            # Longer chunks naturally have more words, so they are more likely to contain query 
            # terms just by chance. BM25 penalises them slightly so they do not always win.
            denominator = query_term_frequency + self.k1 * (
                1 - self.b + self.b * (document_length / self.average_document_length)
            )

            # The score then for this query term is based on how rare the term is, how often it
            # appeared in this chunk, and whether the chunk is unusually long or short.
            term_score = idf * (numerator / denominator)
            score += term_score

            # Used to debug a given term
            logger.debug(
                "BM25 term score | chunk_id=%d | term=%r | tf=%d | idf=%.4f | term_score=%.4f",
                chunk_index,
                term,
                query_term_frequency,
                idf,
                term_score,
            )

        # Used to debug the chunk
        logger.debug("BM25 chunk score | chunk_id=%d | score=%.4f", chunk_index, score)
        
        return score

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:

        if k <= 0:
            logger.warning("invalid k value. k must be > 0")
            return []

        # If we have no chunks, just return the empty list
        if not self.chunks:
            logger.warning("Keyword search called before any chunks were indexed")
            return []
        
        # Make the query a list of words without punctuation and with replacements.
        query_terms = self._tokenise(query)

        if not query_terms:
            logger.warning("Keyword search received an empty query")
            return []

        logger.info(
            "Running keyword search | query=%r | terms=%s | k=%d | chunks=%d",
            query,
            query_terms,
            k,
            len(self.chunks),
        )

        results = []

        # For each chunk, calculate its score and 
        for index, chunk in enumerate(self.chunks):
            score = self._score_chunk(query_terms, index)

            if score > 0:
                results.append({
                    "chunk" : chunk,
                    "score" : score,
                    "retriever" : "keyword",
                    "metadata" : self.metadata[index]
                })
        
        # Sort the results by their score, and put the dictionaries with the 
        # highest score at the front
        results.sort(key=lambda result: result["score"], reverse=True)

        # Return k dictionaries with the highest scores.
        top_results = results[:k]

        logger.info(
            "Keyword search complete | query=%r | matches=%d | returned=%d",
            query,
            len(results),
            len(top_results),
        )

        return top_results
    