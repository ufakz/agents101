import os
from dotenv import load_dotenv
import argparse

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

dotenv_path = os.path.join(parent_dir, ".env")
load_dotenv(dotenv_path)

# Set the token for Hugging Face API usage
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

from smolagents import LiteLLMModel

model = LiteLLMModel(
        model_id="ollama_chat/gemma3:4b", 
        api_base="http://127.0.0.1:11434",
        num_ctx=8192,
)

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, VisitWebpageTool, FinalAnswerTool, Tool, tool

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)

    return best_service

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic gotham').",
        }
    }

    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }

        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic gotham'.")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Alfred - Your AI Butler Assistant")
    parser.add_argument("task", type=str, help="The task you want Alfred to perform")
    args = parser.parse_args()

    # Create and run the agent
    agent = CodeAgent(
        tools=[
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
            suggest_menu,
            catering_service_tool,
            SuperheroPartyThemeTool()
        ],
        model=model,
        max_steps=10,
        verbosity_level=2
    )

    # Run the agent with the provided task
    agent.run(args.task)

if __name__ == "__main__":
    main()