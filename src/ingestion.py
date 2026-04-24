import fitz  # PyMuPDF
import re
import yaml
import os
import argparse
from pathlib import Path

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_text(text: str) -> str:
    """
    Cleans extracted text by stripping excessive whitespace and unnecessary spaces.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Replace more than 2 newlines with 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str, min_paragraph_length: int = 20) -> list:
    """
    Extracts text from a given PDF using PyMuPDF and preserves structural metadata.
    Returns a list of dictionaries with text and metadata.
    """
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num, page in enumerate(doc):
        # Extract blocks of text which preserves paragraph structure relatively well
        blocks = page.get_text("blocks")
        page_text = ""
        
        for b in blocks:
            text = b[4]
            if len(text.strip()) > min_paragraph_length:
                cleaned_block = clean_text(text)
                page_text += cleaned_block + "\n\n"

        pages_data.append({
            "text": page_text.strip(),
            "metadata": {
                "source": os.path.basename(pdf_path),
                "page": page_num + 1
            }
        })
    doc.close()
    return pages_data

def sanitize_filename(name: str):
    return re.sub(r'[^\w\-_\. ]', '_', name)

def process_and_save(pdf_path: str, config: dict):
    os.makedirs(config["paths"]["processed_text_dir"], exist_ok=True)
    min_len = config.get("ingestion", {}).get("min_paragraph_length", 20)
    
    print(f"Extracting text from: {pdf_path}")
    pages_data = extract_text_from_pdf(pdf_path, min_len)
    
    filename_base = sanitize_filename(os.path.basename(pdf_path).replace(".pdf", ""))
    output_path = os.path.join(config["paths"]["processed_text_dir"], f"{filename_base}_extracted.yaml")
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(pages_data, f, allow_unicode=True)
        
    print(f"Successfully extracted {len(pages_data)} pages. Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NCERT PDF and extract text.")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file", required=True)
    args = parser.parse_args()
    
    config = load_config()
    process_and_save(args.pdf, config)
