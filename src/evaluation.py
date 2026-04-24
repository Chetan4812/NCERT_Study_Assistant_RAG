import yaml
import json
import os
import evaluate
from datasets import Dataset
import pandas as pd
from sentence_transformers import util

try:
    from src.rag_pipeline import RAGPipeline
    from src.embeddings import get_embedding_model
except ImportError:
    from rag_pipeline import RAGPipeline
    from embeddings import get_embedding_model

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class Evaluator:
    def __init__(self, rag_pipeline: RAGPipeline, config=None):
        if config is None:
            self.config = load_config()
        else:
            self.config = config
        self.rag = rag_pipeline
        self.embedding_model = get_embedding_model()
        
        # Load HuggingFace metrics
        self.exact_match_metric = evaluate.load("exact_match")
        # Load rouge for generalized token overlap/F1 approximation
        self.rouge_metric = evaluate.load("rouge")

    def create_synthetic_qa(self, chunks_yaml_path: str, output_name: str, num_q_per_chunk: int = 1):
        """Generates synthetic QA pairs given textbook chunks."""
        # Due to API/LLM generation limitations running in batch locally, 
        # normally we'd prompt the local LLM sequentially to make QA. 
        # Here we lay the functional structure.
        print("Synthetic QA data generation requires LLM running and can be slow. Skipping automated full-generation for now.")
        print("You can manually structure a QA dataset JSON in data/qa_dataset/")
        
        sample_qa = [
            {
                "question": "What is the primary topic discussed in the first page?",
                "expected_answer": "This is a placeholder for actual expected answers."
            }
        ]
        
        os.makedirs(self.config["paths"]["qa_dataset_dir"], exist_ok=True)
        path = os.path.join(self.config["paths"]["qa_dataset_dir"], output_name)
        with open(path, "w") as f:
            json.dump(sample_qa, f, indent=4)
        return path

    def run_evaluation(self, test_dataset_path: str, output_report_name: str):
        print(f"Running evaluation on {test_dataset_path}...")
        with open(test_dataset_path, "r") as f:
            qa_data = json.load(f)
            
        results = []
        
        for item in qa_data:
            question = item["question"]
            expected = item["expected_answer"]
            
            # 1. Retrieve & Generate
            response = self.rag.answer(question)
            generated_ans = response["answer"]
            
            # 2. Compute Match Metrics
            em_score = self.exact_match_metric.compute(
                predictions=[generated_ans], 
                references=[expected]
            )["exact_match"]
            
            rouge_scores = self.rouge_metric.compute(
                predictions=[generated_ans],
                references=[expected]
            )
            # We take Rouge-1 F1 as token overlap measure
            f1_score = rouge_scores["rouge1"]
            
            # 3. Compute Semantic Similarity using BGE embeddings
            emb_gen = self.embedding_model.embed_query(generated_ans)
            emb_exp = self.embedding_model.embed_query(expected)
            
            sem_sim = float(util.pytorch_cos_sim(emb_gen, emb_exp)[0][0])
            
            results.append({
                "question": question,
                "expected_answer": expected,
                "generated_answer": generated_ans,
                "exact_match": em_score,
                "f1_score": f1_score,
                "semantic_similarity": sem_sim,
                "sources_retrieved": response["sources"]
            })
            
        df = pd.DataFrame(results)
        
        metrics_summary = {
            "avg_exact_match": df["exact_match"].mean(),
            "avg_f1_score": df["f1_score"].mean(),
            "avg_semantic_similarity": df["semantic_similarity"].mean()
        }
        
        print("\nEvaluation Complete. Average Metrics:")
        print(json.dumps(metrics_summary, indent=4))
        
        # Save report
        os.makedirs(self.config["paths"]["evaluation_reports_dir"], exist_ok=True)
        out_path = os.path.join(self.config["paths"]["evaluation_reports_dir"], output_report_name)
        
        df.to_csv(out_path.replace(".json", ".csv"), index=False)
        with open(out_path, "w") as f:
            json.dump({
                "summary": metrics_summary,
                "detailed_results": results
            }, f, indent=4)
            
        print(f"Report saved to {out_path} and {out_path.replace('.json', '.csv')}")
        return metrics_summary
