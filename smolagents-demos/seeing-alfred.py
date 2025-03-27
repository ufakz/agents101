import os
from dotenv import load_dotenv
import argparse
from PIL import Image
from transformers import pipeline
from smolagents import (
    LiteLLMModel,
    CodeAgent,
    DuckDuckGoSearchTool,
    Tool,
)

# Load environment variables
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
dotenv_path = os.path.join(parent_dir, ".env")
load_dotenv(dotenv_path)
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize the model
model = LiteLLMModel(
    model_id="ollama_chat/gemma3:4b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

class ImageAnalysisTool(Tool):
    name = "image_analysis"
    description = "Analyzes an image and provides a detailed description"

    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the image file to analyze",
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        
    def forward(self, image_path: str):
        try:
            image = Image.open(image_path)
            result = self.pipeline(image)
            return result[0]['generated_text']
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Seeing Alfred - Your AI Butler with Vision")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    # Create agent with image analysis and search capabilities
    agent = CodeAgent(
        tools=[
            ImageAnalysisTool(),
            DuckDuckGoSearchTool(),
        ],
        model=model,
        max_steps=10,
        verbosity_level=2
    )

    prompt = f"""
    1. First, analyze the image at {args.image_path} using image_analysis tool
    2. Based on what you see, search for relevant information
    3. Provide a detailed response combining both the visual analysis and found information
    """

    # Run the agent
    response = agent.run(prompt)
    print("\nAlfred's Response:", response)

if __name__ == "__main__":
    main()
