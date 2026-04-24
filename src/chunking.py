import yaml
import os
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_chunks(extracted_data: list, config: dict):
    """
    Takes the extracted page data and chunks it semantically.
    """
    chunk_config = config.get("chunking", {})
    chunk_size = chunk_config.get("chunk_size", 500)
    chunk_overlap = chunk_config.get("chunk_overlap", 100)
    separators = chunk_config.get("separators", ["\n\n", "\n", ". ", " ", ""])
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len
    )
    
    all_chunks = []
    chunk_id_counter = 0
    
    for page in extracted_data:
        text = page["text"]
        metadata = page["metadata"]
        
        # Split text into chunks
        page_chunks = text_splitter.split_text(text)
        
        for c_text in page_chunks:
            if len(c_text.strip()) == 0:
                continue
            
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = chunk_id_counter
            chunk_id_counter += 1
            
            all_chunks.append({
                "page_content": c_text,
                "metadata": chunk_metadata
            })
            
    return all_chunks

def process_and_save_chunks(extracted_yaml_path: str, config: dict):
    os.makedirs(config["paths"]["chunks_dir"], exist_ok=True)
    
    print(f"Loading extracted text from: {extracted_yaml_path}")
    with open(extracted_yaml_path, "r", encoding="utf-8") as f:
        extracted_data = yaml.safe_load(f)
        
    chunks = create_chunks(extracted_data, config)
    
    filename_base = os.path.basename(extracted_yaml_path).replace("_extracted.yaml", "")
    output_path = os.path.join(config["paths"]["chunks_dir"], f"{filename_base}_chunks.yaml")
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(chunks, f, allow_unicode=True)
        
    print(f"Successfully created {len(chunks)} chunks. Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk extracted text from PDF.")
    parser.add_argument("--input", type=str, help="Path to the extracted YAML file", required=True)
    args = parser.parse_args()
    
    config = load_config()
    process_and_save_chunks(args.input, config)
