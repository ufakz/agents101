import os
from dotenv import load_dotenv
import chromadb
import nest_asyncio

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


class AccountantAgent:
    def __init__(self, data_dir: str = "./accounting", db_path: str = "./alfred_chroma_db"):
        self.db_path = db_path
        setup_environment()
        
        reader = SimpleDirectoryReader(input_dir=data_dir)
        documents = reader.load_data()

        # Initialize LLM and embedding model

        # Provide a more capable model for better performance
        # Note: Ensure you have the model downloaded and available in your environment
        self.llm = OpenAI(
            model="gpt-4o",
            system_prompt="""You are a helpful assistant that can perform calculations and search through documents to answer questions.""",
        )
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Setup vector store and query engine
        self.db = chromadb.PersistentClient(path=db_path)
        self.collection = self.db.get_or_create_collection(name="alfred")
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(),
                self.embed_model,
            ],
            vector_store=self.vector_store,
        )
        self.pipeline.run(documents=documents)
        
        # Define the index from the vector store
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, embed_model=self.embed_model
        )
        nest_asyncio.apply()
        self.query_engine = self.index.as_query_engine(llm=self.llm)

        # Create the query engine tool
        self.query_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="accounting_lookup",
            description="Searches through accounting documents and statements to find relevant information",
            return_direct=False,
        )

        # Create the calculator agent with math tools
        self.calculator_agent = ReActAgent(
            name="calculator",
            description="Performs basic arithmetic operations for accounting calculations",
            system_prompt="You are an accounting calculator assistant. Use your tools for any math operation related to accounting.",
            tools=[add, subtract, multiply, divide],
            llm=self.llm,
        )

        # Create the query agent for looking up accounting information
        self.query_agent = ReActAgent(
            name="accounting_info",
            description="Looks up information in accounting documents",
            system_prompt="Use your tool to query accounting documents and statements to answer questions about financial information",
            tools=[self.query_tool],
            llm=self.llm,
        )

        self.root_agent = ReActAgent(
            name="root_agent",
            description="Root agent that coordinates between calculator and query agents",
            system_prompt="You are a root agent that coordinates between calculator and query agents. You should use the appropriate agent to achieve your obectives.",
            llm=self.llm,
        )

        # Create the multi-agent workflow
        self.agent_workflow = AgentWorkflow(
            agents=[self.root_agent, self.calculator_agent, self.query_agent],
            root_agent="root_agent",
        )

    async def process_query(self, query: str, context: Context = None) -> str:
        """Process a query using the multi-agent system"""
        if context is None:
            context = Context(self.agent_workflow)

        response = await self.agent_workflow.run(query, ctx=context)
        return response


def setup_environment():
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dotenv_path = os.path.join(parent_dir, ".env")
    load_dotenv(dotenv_path)
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


async def interactive_mode(agent: AccountantAgent):
    """Run the agent in interactive mode."""
    print("Welcome to the Accountant Agent Interactive Mode!")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            response = await agent.process_query(query)
            print("\nResponse:", response)
        except Exception as e:
            print(f"\nError: {str(e)}")


async def main():
    """Main function to run the AccountantAgent."""
    agent = AccountantAgent()
    await interactive_mode(agent)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
