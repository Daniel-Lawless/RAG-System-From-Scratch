from pathlib import Path
from typing import Any
from vector_search import VectorSearch
from keyword_search import KeywordSearch
from hybrid_search import HybridSearch
from chunking import Chunking
from embeddings import Embeddings
from logging_config import configure_logging

import json
import logging

logger=logging.getLogger(__name__)

class RetrievalEvaluator:

    def __init__(
            self,
            index_path: Path = Path("storage/index"),
            eval_path: Path = Path("eval"),
            k: int = 10
            ):
        
        self.index_path = Path(index_path)
        self.eval_path = Path(eval_path)
        self.k = k

        # Load these once, then reuse them for every question.
        self.embedder = Embeddings()
        self.chunker = Chunking()

        # Populate vector_search
        self.vector_search = VectorSearch.from_index(self.index_path)

        # Populate keyword search
        self.keyword_search = KeywordSearch.from_jsonl(self.index_path)

        # Populate hybridsearch
        self.hybrid_search = HybridSearch(
            vector_search=self.vector_search,
            keyword_search=self.keyword_search,
            embedder=self.embedder
        )
    
    # Load evaluation questions
    def _load_questions(self) -> list[dict[str,Any]]:
        question_file = self.eval_path / "questions.json"
        with question_file.open("r", encoding="utf-8") as file:
            return json.load(file)

    # Run and obtain results from vector search
    def _run_vector_search(self, query: str) -> list[dict[str, Any]]:

        # Embed the query
        query_embedding = self.embedder.embed(query)

        return self.vector_search.search(
            query_embedding=query_embedding,
            k=self.k
        )
    
    # Run and obtain results from keyword search
    def _run_keyword_search(self, query: str) -> list[dict[str, Any]]:
        return self.keyword_search.search(
            query=query,
            k=self.k
        )
    
    # Run and obtain results from hybrid search
    def _run_hybrid_search(self, query: str) -> list[dict[str, Any]]:
        return self.hybrid_search.search(
            query=query,
            k=self.k
        )

    # Answers how close to the top the first correct source file appears.
    def _reciprocal_rank(
            self,
            results : list[dict[str, Any]],
            relevant_sources: list[str]
            ) -> float:
        
        # Assign a rank to each result
        for rank, result in enumerate(results, start=1):
            # extract the source file the retriever used for this record
            source_file = result["metadata"]["source_file"]

            # higher value if it chose a record closer to the start with the relevant source
            if source_file in relevant_sources:
                return 1 / rank
        
        # If it didn't retrieve a single record that used the relevant file, it scores 0.
        return 0.0
    
    # Answers how many of the top k retrieved chunks were from the correct source file.
    def _correct_source_count_at_k(
            self,
            results : list[dict[str, Any]],
            relevant_sources : list[str]
            ) -> int:
        
        # Initialise to 0
        number_of_hits = 0
        
        for record in results:
            # Extract source file from this record
            source_file = record["metadata"]["source_file"]
            
            # If the source_file is in relevant_sources, the retriever got a hit, so add 1.
            if source_file in relevant_sources:
                number_of_hits += 1

        return number_of_hits
    
    # Answers whether the retriever retrieved even one chunk with the expected sourcefile
    def _hit_at_k(
            self,
            results : list[dict[str, Any]],
            relevant_sources : list[str]
            ) -> int:
        
        for record in results:
            # Extract source file from this record
            source_file = record["metadata"]["source_file"]
            
            # If the source_file is in relevant_sources, the retriever got a hit, so return 1.
            if source_file in relevant_sources:
                return 1

        return 0

    def _evaluate_retriever(
            self,
            retriever_name : str,
            questions: list[dict[str, Any]],
            ) -> tuple[dict[str, float], list[dict[str, Any]]]:

        list_of_results = []
        
        total_hits = 0
        total_correct_hits: int = 0
        total_reciprocal_rank = 0.0

        for question_record in questions:
            # Retrieve the question and relevant_sources
            question = question_record["question"]
            relevant_sources = question_record["relevant_sources"]

            # note, a trick here is to name each result set "results", that way regardless
            # of which one runs, the following code will run with the correct result set.
            if retriever_name == "vector":
                results = self._run_vector_search(query=question)
            elif retriever_name == "keyword":
                results = self._run_keyword_search(query=question)
            elif retriever_name == "hybrid":
                results= self._run_hybrid_search(query=question)
            else: 
                raise ValueError("Unknown retriever.")

            # Calculate hit per question
            hit = self._hit_at_k(
                results=results,
                relevant_sources=relevant_sources
            )

            # Calculate how many of the k results were from the correct source_file
            number_of_correct_hits = self._correct_source_count_at_k(
                results=results,
                relevant_sources=relevant_sources
            )

            # Calculate reciprocal rank per question
            reciprocal_rank = self._reciprocal_rank(
                results=results,
                relevant_sources=relevant_sources
            )

            # Sum up each hit to give total hit
            total_hits += hit

            # Sum up number of correct hits per question to give total correct hits.
            total_correct_hits += number_of_correct_hits

            # Proportion of the top k results that came from the correct source.
            precision = number_of_correct_hits / self.k

            # Sum up each reciprocal rank to give total reciprocal rank
            total_reciprocal_rank += reciprocal_rank

            retrieved_sources = []

            # Calculate each chunks returned contents 
            for rank, result in enumerate(results, start=1):
                metadata = result["metadata"]

                record = {
                    "rank" : rank,
                    "source_file" : metadata["source_file"],
                    "chunk_index" : metadata["chunk_index"],
                    "score" : round(result["score"], 4),
                    "retriever" : result["retriever"]
                }

                retrieved_sources.append(record)

            # Combine this with the metrics for the whole question.
            result_for_question = {
                "question" : question,
                "relevant_sources" : relevant_sources,
                "metrics" : {
                    "hit" : hit,
                    "number_of_correct_hits" : number_of_correct_hits,
                    "precision" : precision,
                    "reciprocal_rank" : reciprocal_rank
                },
                "retrieved_sources" : retrieved_sources
            }

            # append to our list of combined results
            list_of_results.append(result_for_question)        
        
        number_of_questions = len(questions)

        # Average the metrics
        mean_hit_at_k = total_hits / number_of_questions
        mean_correct_source_count_at_k = total_correct_hits / number_of_questions
        mean_precision_at_k = total_correct_hits / (self.k * number_of_questions)
        mean_reciprocal_rank = total_reciprocal_rank / number_of_questions

        return ({
            f"hit@{self.k}": mean_hit_at_k,
            f"mean_correct_source_count@{self.k}": mean_correct_source_count_at_k,
            f"precision@{self.k}": mean_precision_at_k,
            f"mrr@{self.k}": mean_reciprocal_rank,
        }, list_of_results)

    def run(self) -> None:

        logger.info("Starting evaluation...")

        # Load the questions from eval/questions.json
        questions = self._load_questions()

        if not questions:
            raise ValueError("No evaluation questions found.")

        retrievers = ["vector", "keyword", "hybrid"]

        results = {} 

        results_per_question_retriever = {}

        # For each retriever
        for retriever in retrievers:

            logger.info("Evaluating %s retriever", retriever)

            # Extract overall metrics and per question metrics of the given retriever
            metrics, per_question_metrics = self._evaluate_retriever(
                retriever_name=retriever,
                questions=questions
            )

            # Populate results dictionary for overall results
            results[retriever] = {
                f"mean_hit@{self.k}": round(metrics[f"hit@{self.k}"], 4),
                f"mean_correct_source_count@{self.k}": round(metrics[f"mean_correct_source_count@{self.k}"], 4),
                f"mean_precision@{self.k}": round(metrics[f"precision@{self.k}"], 4),
                f"mean_reciprocal_rank@{self.k}": round(metrics[f"mrr@{self.k}"], 4),
            }

            # Populate dictionary for per questions results
            results_per_question_retriever[retriever] = per_question_metrics

        # Define paths
        results_path = self.eval_path / "results.json"
        per_question_results_path = self.eval_path / "per_question_results.json"

        # Write overall results to results.json
        logger.info("Creating %s", results_path)
        with results_path.open("w") as file:
            json.dump(results, file, indent=4)
        
        # Write per question results to per_question_results.json
        logger.info("Creating %s", per_question_results_path)
        with per_question_results_path.open("w") as file:
            json.dump(results_per_question_retriever, file, indent=4)

        logger.info("Evaluation complete")

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)

    noisy_loggers = [
        "vector_search",
        "keyword_search",
        "hybrid_search",
        "sentence_transformers",
        "httpx",
        "httpcore",
        "huggingface_hub",
        "transformers",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

if __name__ == "__main__":
    configure_logging()

    evaluator = RetrievalEvaluator(k=4)
    evaluator.run()
    evaluator = RetrievalEvaluator(k=4)
    evaluator.run()
