import os
import argparse
from dotenv import load_dotenv
import chromadb
import nest_asyncio

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

class RAGSystem:
    def __init__(self, db_path: str = "./alfred_chroma_db"):
        self.db_path = db_path
        self.db = chromadb.PersistentClient(path=db_path)
        self.collection = self.db.get_or_create_collection(name="alfred")
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.query_engine = None
        self.initialized = False
        setup_environment()
        
    def initialize_from_documents(self, data_dir: str):
        """Initialize the vector store with documents"""
        reader = SimpleDirectoryReader(input_dir=data_dir)
        documents = reader.load_data()

        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(),
                embed_model,
            ],
            vector_store=self.vector_store,
        )

        pipeline.run(documents=documents)
        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, embed_model=embed_model
        )
        
        nest_asyncio.apply()
        llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
        self.query_engine = index.as_query_engine(
            llm=llm,
            response_mode="tree_summarize",
        )
        
        self.initialized = True
        return "Vector store initialized successfully!"

    def query(self, prompt: str) -> str:
        """Query the initialized system"""
        if not self.initialized:
            return "Error: System not initialized. Please initialize with documents first."
        response = self.query_engine.query(prompt)
        return str(response)

def interactive_mode(rag_system: RAGSystem):
    """Run interactive query mode"""
    print("\nEntering interactive mode. Type 'quit' to exit.")
    while True:
        prompt = input("\nEnter your question: ").strip()
        if prompt.lower() == 'quit':
            break
        response = rag_system.query(prompt)
        print(f"\nResponse: {response}")

def setup_environment():
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dotenv_path = os.path.join(parent_dir, ".env")
    load_dotenv(dotenv_path)
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

def main():
    rag_system = RAGSystem()
    
    parser = argparse.ArgumentParser(description='RAG-based question answering system')
    parser.add_argument('--data-dir', type=str, help='Directory containing the input data')
    
    args = parser.parse_args()

    if not args.data_dir:
        print("Error: --data-dir is required for initialization")
        return
    result = rag_system.initialize_from_documents(args.data_dir)
    print(result)
    
    interactive_mode(rag_system)

if __name__ == "__main__":
    main()