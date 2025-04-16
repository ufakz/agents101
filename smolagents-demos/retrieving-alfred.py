import os
from dotenv import load_dotenv
from smolagents import LiteLLMModel
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, HfApiModel
from pathlib import Path
from typing import List, Union
from PyPDF2 import PdfReader
import argparse

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

dotenv_path = os.path.join(parent_dir, ".env")
load_dotenv(dotenv_path)

# Set the token for Hugging Face API usage
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

model = LiteLLMModel(
        model_id="ollama_chat/gemma3:4b", 
        api_base="http://127.0.0.1:11434",
        num_ctx=8192,
)

def load_documents(file_paths: Union[str, List[str]]) -> List[Document]:
    """Load documents from PDF files."""
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    documents = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF: {file_path}")
            
        pdf_reader = PdfReader(str(path))
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page": page_num + 1
                    }
                ))
    return documents

class RetrieverTool(Tool):
    name = "document_retriever"
    description = "Uses semantic search to retrieve information from the documents to answer the specified query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be a query related to the documents attached.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=5  # Retrieve the top 5 documents
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved information from document:\n" + "".join(
            [
                f"\n " + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

def main():
    parser = argparse.ArgumentParser(description='Document Retrieval System')
    parser.add_argument('--query', type=str, required=True,
                        help='The query you want to perform (e.g., "get the names of the famous one")')
    parser.add_argument('--files', nargs='+', required=True,
                      help='Paths to PDF files containing party planning ideas')
    
    args = parser.parse_args()
    
    try:
        
        docs = load_documents(args.files)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        docs_processed = text_splitter.split_documents(docs)
        
        retriever = RetrieverTool(docs_processed)
        
        agent = CodeAgent(tools=[retriever], model=model)
        
        response = agent.run(
            task=args.query,
        )
        
        print(response)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()