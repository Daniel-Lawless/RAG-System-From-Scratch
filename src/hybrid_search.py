import logging
from typing import Any

from vector_search import VectorSearch
from keyword_search import KeywordSearch
from embeddings import Embeddings

logger = logging.getLogger(__name__)

class HybridSearch:

    # Default behaviour is standard RRF
    def __init__(
            self,
            vector_search: VectorSearch,
            keyword_search: KeywordSearch,
            embedder: Embeddings,
            vector_weight: float = 1,
            keyword_weight: float = 1,
            rank_constant: int = 60
            ):
        
        self.vector_search = vector_search
        self.keyword_search = keyword_search
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rank_constant = rank_constant

    def _record_key(self, record: dict[str,Any]) -> tuple[str, int]:
        metadata = record["metadata"]
        return (metadata["source_file"], metadata["chunk_index"])
    
    def _add_ranked_results(
            self,
            fused_results: dict[tuple[str,int], dict[str,Any]],
            results: list[dict[str, Any]],
            retriever_name: str,
            weight:float
            ) -> None :
        
        for rank, record in enumerate(results, start=1):
            key = self._record_key(record)

            # Weighted Reciprocal Rank Fusion score
            # The rank_constant, defualt 60, prevents rank 1 from completely
            # dominating everything else. It smooths the differences between ranks.
            rrf_score = weight * (1 / (self.rank_constant + rank))

            # If this result is already in fused results from vector search, 
            # don't add it again when going through keyword search 
            if key not in fused_results:
                fused_results[key] = {
                    "chunk" : record["chunk"],
                    "score" : 0.0,
                    "retriever" : "hybrid",
                    "metadata" : record["metadata"],
                    # Used for debugging
                    "retriever_information" : {}
                }
            
            # The reason this is done outside the dictionary is because if both 
            # vector search and keyword search pick this record, both contribute
            # to its score
            fused_results[key]["score"] += rrf_score

            # Debigging info to find out what retriever it came from, its rank, 
            # and its original score.
            fused_results[key]["retriever_information"][retriever_name] = {
                "rank" : rank,
                "raw_score" : record["score"]
            }
    
    def search(self, query: str, k: int) -> list[dict[str, Any]]:

        if k <= 0:
            logger.warning("Inavlid value for k. k must be > 0")
            return []
        
        if self.vector_weight < 0 or self.keyword_weight < 0:
            raise ValueError("Retriever weights cannot be negative")

        if self.vector_weight == 0 and self.keyword_weight == 0:
            raise ValueError("At least one retriever weight must be greater than 0")
        
        logger.info("Starting hybrid search...")
    
        # Retrieve more candidates then we finally return.
        # This gives retrievers more room to contribute useful chunks
        candidate_k = max(k * 3, 20)

        query_embedding = self.embedder.embed(query)

        # Retrieve best results via vector search
        vector_results = self.vector_search.search(
            query_embedding=query_embedding,
            k=candidate_k
        )

        # Retrieve best results via keyword search
        keyword_results = self.keyword_search.search(
            query=query,
            k=candidate_k
        )

        # Dictionary to hold the combined results.
        fused_results = {}

        # populate fused results and calculate scores
        # from vector_search
        self._add_ranked_results(
            fused_results=fused_results,
            results=vector_results,
            retriever_name="vector",
            weight=self.vector_weight
        )

        # populate fused results and calculate scores
        # from keyword_search
        self._add_ranked_results(
            fused_results=fused_results,
            results=keyword_results,
            retriever_name="keyword",
            weight=self.keyword_weight
        )

        # Get the values of the results only, this is {
                #     "chunk" : buffons needle is...,
                #     "score" : 0.0,
                #     "retriever" : "hybrid",
                #     "metadata" : {...},
                #      Used for debugging
                #     "retriever_scores" : {
                #                           "vector": {
                #                                       "rank" : 1,
                #                                       "raw_score" : 7.8
                #                                       }
                #                           "keyword:" {
                #                                       "rank" : 4,
                #                                       "raw_score" : 3.3
                #                                       }
                #                            }
                # } (if chunk retrieved from both vector search and keyword search)
        results = list(fused_results.values())

        for result in results:
            # If record is retrieved from both vector search and keyword search, 
            # the rder is guranteed to be "vector", then "keyword" so no need to sort for 
            # consistent output.
            retrievers = result["retriever_information"].keys()
            result["retriever"] = "hybrid(" + "+".join(retrievers) + ")"

        # Sort the records by score
        results.sort(key = lambda x : x["score"], reverse=True)
        
        # Return the top k record with the highest score
        top_k_results = results[:k]

        logger.info(
            "Hybrid search complete | vector_matches=%d | keyword_matches=%d | returned=%d",
            len(vector_results),
            len(keyword_results),
            len(top_k_results)
        )

        return top_k_results
