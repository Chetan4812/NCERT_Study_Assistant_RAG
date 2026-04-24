import yaml
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

try:
    from src.vector_store import VectorDBManager
except ImportError:
    from vector_store import VectorDBManager

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class RAGPipeline:
    def __init__(self, index_name: str, config=None):
        if config is None:
            self.config = load_config()
        else:
            self.config = config
            
        self.vector_manager = VectorDBManager(self.config)
        self.vector_manager.load_index(index_name)
        
        self._init_llm()
        self._init_prompt()

    def _init_llm(self):
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "ollama")
        model_name = llm_config.get("model_name", "llama3")
        temperature = llm_config.get("temperature", 0.1)

        if provider == "ollama":
            # Attempt to use ollama. If not running, it will throw an error when called.
            self.llm = Ollama(model=model_name, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _init_prompt(self):
        template = """System Prompt:
You are an expert NCERT tutor. Answer strictly based on the provided context. Do not hallucinate. If the answer is not present, say 'Not found in the provided text.' Give clear, student-friendly explanations.

Context:
{retrieved_chunks}

Question:
{user_query}

Answer:"""
        self.prompt = PromptTemplate(
            input_variables=["retrieved_chunks", "user_query"],
            template=template
        )

    def retrieve(self, query: str):
        top_k = self.config.get("retrieval", {}).get("top_k", 4)
        results = self.vector_manager.search(query, top_k=top_k)
        
        # Results is list of (Document, score)
        # We need to extract the text content
        context_texts = []
        sources = []
        for doc, score in results:
            context_texts.append(doc.page_content)
            # Gather sources for UI / transparency
            pg = doc.metadata.get("page", "?")
            sources.append(f"Page {pg}")
            
        return "\n\n---\n\n".join(context_texts), list(set(sources)), results

    def answer(self, query: str):
        context_str, sources, raw_results = self.retrieve(query)
        
        # Build prompt
        final_prompt = self.prompt.format(
            retrieved_chunks=context_str,
            user_query=query
        )
        
        # Generate answer
        try:
            response = self.llm.invoke(final_prompt)
        except Exception as e:
            response = f"LLM Generation Error: Ensure {self.config.get('llm', {}).get('provider')} backend is running. Details: {str(e)}"
            
        return {
            "answer": response.strip(),
            "sources": sources,
            "context": context_str,
            "raw_results": raw_results
        }
