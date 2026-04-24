import os
import yaml
import argparse
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

try:
    from src.embeddings import get_embedding_model
except ImportError:
    from embeddings import get_embedding_model

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class VectorDBManager:
    def __init__(self, config=None):
        if config is None:
            self.config = load_config()
        else:
            self.config = config
        
        self.vector_db_dir = self.config["paths"]["vector_db_dir"]
        os.makedirs(self.vector_db_dir, exist_ok=True)
        self.embedding_model = get_embedding_model()
        self.index = None

    def build_from_chunks(self, chunks_yaml_path: str):
        """Builds a FAISS index from a chunks yaml file."""
        print(f"Loading chunks from: {chunks_yaml_path}")
        with open(chunks_yaml_path, "r", encoding="utf-8") as f:
            chunks_data = yaml.safe_load(f)
            
        documents = []
        for c in chunks_data:
            documents.append(
                Document(page_content=c["page_content"], metadata=c["metadata"])
            )
        
        print(f"Building FAISS vector index from {len(documents)} logic. This may take a moment...")
        self.index = FAISS.from_documents(documents, self.embedding_model)
        
        filename_base = os.path.basename(chunks_yaml_path).replace("_chunks.yaml", "")
        save_path = os.path.join(self.vector_db_dir, filename_base)
        
        self.index.save_local(save_path)
        print(f"Vector Index saved locally at: {save_path}")
        return save_path

    def load_index(self, index_name: str):
        """Loads a pre-built FAISS index from disk."""
        load_path = os.path.join(self.vector_db_dir, index_name)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No vector index found at {load_path}")
            
        print(f"Loading FAISS index from {load_path}")
        self.index = FAISS.load_local(load_path, self.embedding_model, allow_dangerous_deserialization=True)
        return self.index
        
    def search(self, query: str, top_k: int = 4):
        """Searches the index and returns top_k results."""
        if not self.index:
            raise ValueError("Index is not loaded or built yet.")
        
        # similarity_search_with_score returns List[Tuple[Document, float]]
        results = self.index.similarity_search_with_score(query, k=top_k)
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create FAISS Vector Store from chunks.")
    parser.add_argument("--input", type=str, help="Path to the chunks YAML file", required=True)
    args = parser.parse_args()
    
    manager = VectorDBManager()
    manager.build_from_chunks(args.input)
