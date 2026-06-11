from pathlib import Path
from typing import Any
import json

from vector_search import VectorSearch
from keyword_search import KeywordSearch
from hybrid_search import HybridSearch
from chunking import Chunking
from embeddings import Embeddings


class RetrievalEvaluator:

    def __init__(
            self,
            index_path: Path = Path("storage/index"),
            eval_path: Path = Path("eval/questions.json"),
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
        with self.eval_path.open("r", encoding="utf-8") as file:
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
            
            # If the source_file is in relevant_sources, the retriever got a hit, so add 1.
            if source_file in relevant_sources:
                return 1

        return 0

    def _evaluate_retriever(
            self,
            retriever_name : str,
            questions: list[dict[str, Any]],
            ) -> dict[str, float]:
        
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

            print()
            print(f"Question : {question}")
            print(f"retriever : {retriever_name}")
            print(f"hit@{self.k} : {hit}")
            print(f"correct_source_count@{self.k} : {number_of_correct_hits}")
            print(f"precision@{self.k} : {precision:.4f}")
            print(f"RR@{self.k} : {reciprocal_rank:.4f}")
            print("retrieved sources:")

            for rank, result in enumerate(results, start=1):
                metadata = result["metadata"]
                print(
                    f"{rank}. {metadata['source_file']}",
                    f"chunk = {metadata['chunk_index']}",
                    f"score = {result['score']:.4f}",
                    f"retriever = {result['retriever']}"
                )
        
        number_of_questions = len(questions)

        # Average the metrics
        mean_hit_at_k = total_hits / number_of_questions
        mean_correct_source_count_at_k = total_correct_hits / number_of_questions
        mean_precision_at_k = total_correct_hits / (self.k * number_of_questions)
        mean_reciprocal_rank = total_reciprocal_rank / number_of_questions

        return {
            f"hit@{self.k}": mean_hit_at_k,
            f"correct_source_count@{self.k}": mean_correct_source_count_at_k,
            f"precision@{self.k}": mean_precision_at_k,
            f"mrr@{self.k}": mean_reciprocal_rank,
        }

    # Run the evaluator for each retriever.
    def run(self) -> None:
        # Load the questions from eval/questions.json
        questions = self._load_questions()

        if not questions:
            raise ValueError("No evaluation questions found.")

        retrievers = ["vector", "keyword", "hybrid"]

        print()
        print(f"running retrieval evaluation with k = {self.k} ")
        print("=" * 60)

        for retriever in retrievers:
            metrics = self._evaluate_retriever(
                retriever_name=retriever,
                questions=questions
                )
            
            print("=" * 60)
            print(f"summary for retriever {retriever}:")
            print(
                f"hit@{self.k} : {metrics[f'hit@{self.k}']:.4f}\n"
                f"correct_source_count@{self.k} : {metrics[f'correct_source_count@{self.k}']:.4f}\n"
                f"precision@{self.k} : {metrics[f'precision@{self.k}']:.4f}\n"
                f"mrr@{self.k} : {metrics[f'mrr@{self.k}']:.4f}"
            )
            print("=" * 60)

if __name__ == "__main__":
    evaluator = RetrievalEvaluator(k=4)
    evaluator.run()
