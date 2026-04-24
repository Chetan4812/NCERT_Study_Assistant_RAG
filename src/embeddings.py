from langchain_huggingface import HuggingFaceEmbeddings
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_embedding_model():
    """
    Initializes and returns the BGE HuggingFace embedding model.
    """
    config = load_config()
    emb_config = config.get("embeddings", {})
    
    model_name = emb_config.get("model_name", "BAAI/bge-large-en-v1.5")
    normalize = emb_config.get("normalize_embeddings", True)
    import torch
    
    device = emb_config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but PyTorch was not compiled with CUDA support. Falling back to CPU for embeddings.")
        device = "cpu"
        
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': normalize} # set True to compute cosine similarity
    
    print(f"Loading embedding model: {model_name} on {device}")
    
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    
    return hf_embeddings
