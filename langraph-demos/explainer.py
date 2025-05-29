import os
import argparse
import base64
import requests
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
from langdetect import detect
from duckduckgo_search import DDGS
import ollama

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path)

os.environ["OPENAI_API_KEY"] = os.getenv(
    "OPENAI_API_KEY"
)  # Handled by ChatOpenAI if OPENAI_API_KEY is in env

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class ExplainerAgentState(TypedDict):
    image_path: str
    extracted_text: Optional[str]
    is_english: Optional[bool]
    text_for_search_and_explainer: Optional[str]
    explainer_content: Optional[str]
    final_output_messages: list[str]
    error_message: Optional[str]
    was_translated: Optional[bool]


### --- Tools Definitions ---
def vlm_extract_text_from_image(image_path: str) -> Optional[str]:
    """Extracts text from an image file using a multimodal model."""
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Extract all the text from this image. Return only the extracted text, no explanations or apologies if no text is found. If no text, return an empty string.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ]
            )
        ]
        response = llm.invoke(message)
        extracted = response.content.strip()
        return extracted if extracted else None
    except Exception as e:
        print(f"Error during VLM text extraction for {image_path}: {e}")
        return None


def is_english(text: str) -> bool:
    """Check if text is English using langdetect library."""
    if not text or len(text.strip()) < 3:
        return False
    try:
        detected_lang = detect(text)
        return detected_lang == "en"
    except Exception as e:
        print(f"Error detecting language: {e}")
        # Fallback to simple heuristic
        common_words = ["the", "is", "are", "and", "a", "in", "it", "to", "of", "for"]
        return any(word in text.lower() for word in common_words)


def translate_to_english_ollama(text: str) -> str:
    """Translate text to English using Ollama local LLM."""
    try:
        res = ollama.generate(
            model="gemma3:4b",
            prompt=f"Translate the following text to English. Return ONLY the translated text, no explanations or additional text: {text}",
            stream=False,  # Set to True if you want streaming responses
        )
        
        if res.done:
            translated_text = res.response.strip()
            return translated_text if translated_text else text
        else:
            print(f"Ollama translation failed with status {response.status_code}")
            return text

    except Exception as e:
        print(f"Error during Ollama translation: {e}")
        print("Falling back to OpenAI for translation...")

        try:
            message = [
                HumanMessage(
                    content=f"Translate the following text to English. Return ONLY the translated text:\n\n{text}"
                )
            ]
            response = llm.invoke(message)
            return response.content.strip()
        except Exception as fallback_error:
            print(f"Fallback translation also failed: {fallback_error}")
            return f"[Translation failed]: {text}"


def web_search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Perform web search using DuckDuckGo."""
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No search results found for query: '{query}'"

        # Format results into a readable summary
        formatted_results = []
        for i, result in enumerate(results[:max_results], 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            url = result.get("href", "No URL")
            formatted_results.append(f"{i}. {title}\n   {body}\n   Source: {url}")

        return f"Search results for '{query}':\n\n" + "\n\n".join(formatted_results)

    except Exception as e:
        print(f"Error during web search: {e}")
        return f"Web search failed for query: '{query}'. Error: {str(e)}"


### --- Node Functions ---
def extract_text_node(state: ExplainerAgentState) -> ExplainerAgentState:
    print(f"Node: Extracting text from {state['image_path']}...")
    extracted = vlm_extract_text_from_image(state["image_path"])
    if not extracted:
        state["error_message"] = (
            f"No text found or error extracting text from {state['image_path']}."
        )
        state["extracted_text"] = None
    else:
        state["extracted_text"] = extracted
        state["error_message"] = None
    return state


def language_processing_node(state: ExplainerAgentState) -> ExplainerAgentState:
    print("Node: Processing language...")
    if not state["extracted_text"]:  # Should not happen if routed correctly
        state["error_message"] = "Cannot process language: No extracted text."
        return state

    is_eng = is_english(state["extracted_text"])
    state["is_english"] = is_eng
    state["was_translated"] = False

    if is_eng:
        state["text_for_search_and_explainer"] = state["extracted_text"]
    else:
        translated_text = translate_to_english_ollama(state["extracted_text"])
        state["text_for_search_and_explainer"] = translated_text
        state["was_translated"] = True
    return state


def generate_explainer_node(state: ExplainerAgentState) -> ExplainerAgentState:
    print("Node: Generating explainer...")
    if not state["text_for_search_and_explainer"]:
        state["error_message"] = "Cannot generate explainer: No text to process."
        return state

    search_query = state["text_for_search_and_explainer"][
        :200
    ]  # Use a snippet for search
    search_results_summary = web_search_duckduckgo(search_query)

    prompt_template = f"""You are an AI assistant. You have been provided with text, possibly extracted from an image.
The text is: "{state['text_for_search_and_explainer']}"
Web search results related to this text: "{search_results_summary}"

Your task is to provide a concise explainer about the main topic of the provided text.
Use the web search results to enrich your explanation. The explainer should be easy to understand.
Explainer:
"""
    response = llm.invoke([HumanMessage(content=prompt_template)])
    state["explainer_content"] = response.content.strip()
    return state


def prepare_final_output_node(state: ExplainerAgentState) -> ExplainerAgentState:
    print("Node: Preparing final output...")
    messages = []
    messages.append(f"--- Results for: {state['image_path']} ---")
    messages.append(f"Original Extracted Text:\\n{state['extracted_text']}\\n")

    if state.get("was_translated"):
        messages.append(
            f"Translated Text (for explainer):\\n{state['text_for_search_and_explainer']}\\n"
        )

    if state["explainer_content"]:
        messages.append(f"Generated Explainer:\\n{state['explainer_content']}")

        # Save explainer to file
        try:
            base_name = os.path.splitext(os.path.basename(state["image_path"]))[0]
            output_dir = os.path.dirname(state["image_path"])
            if not output_dir:  # If only filename was given, save in current dir
                output_dir = "."
            explainer_filename = f"{base_name}_explainer.txt"
            full_explainer_path = os.path.join(output_dir, explainer_filename)

            with open(full_explainer_path, "w", encoding="utf-8") as f:
                f.write(f"Original Extracted Text:\\n{state['extracted_text']}\\n\\n")
                if state.get("was_translated"):
                    f.write(
                        f"Translated Text (for explainer):\\n{state['text_for_search_and_explainer']}\\n\\n"
                    )
                f.write(f"Generated Explainer:\\n{state['explainer_content']}")
            messages.append(f"\\nExplainer saved to: {full_explainer_path}")
        except Exception as e:
            messages.append(f"\\nError saving explainer: {e}")
    else:
        messages.append("No explainer was generated.")
        if state.get("error_message"):  # If an earlier error prevented explainer
            messages.append(f"Reason: {state['error_message']}")

    state["final_output_messages"] = messages
    return state


def prepare_error_output_node(state: ExplainerAgentState) -> ExplainerAgentState:
    print("Node: Preparing error output...")
    messages = [
        f"--- Error processing: {state['image_path']} ---",
        state.get("error_message", "An unknown error occurred."),
    ]
    state["final_output_messages"] = messages
    return state


# --- Graph Definition ---
workflow = StateGraph(ExplainerAgentState)

workflow.add_node("extract_text", extract_text_node)
workflow.add_node("language_process", language_processing_node)
workflow.add_node("generate_explainer", generate_explainer_node)
workflow.add_node("prepare_output", prepare_final_output_node)
workflow.add_node("prepare_error_output", prepare_error_output_node)

workflow.set_entry_point("extract_text")


def should_continue_after_extraction(state: ExplainerAgentState):
    if state.get("error_message"):  # Error during extraction or no text
        return "prepare_error_output"
    return "language_process"


workflow.add_conditional_edges("extract_text", should_continue_after_extraction)


# If language processing itself fails (e.g. no extracted_text somehow)
def should_continue_after_language_processing(state: ExplainerAgentState):
    if state.get("error_message") or not state.get("text_for_search_and_explainer"):
        # If error_message was set by language_process, or if it failed to produce text_for_search_and_explainer
        # We might need to ensure error_message is set if text_for_search_and_explainer is missing
        if not state.get("error_message"):
            state["error_message"] = (
                "Failed to prepare text for explainer during language processing."
            )
        return "prepare_error_output"
    return "generate_explainer"


workflow.add_conditional_edges(
    "language_process", should_continue_after_language_processing
)


def should_continue_after_explainer_gen(state: ExplainerAgentState):
    if state.get("error_message") or not state.get("explainer_content"):
        if not state.get("error_message"):
            state["error_message"] = "Failed to generate explainer content."
        return "prepare_error_output"  # Use error output if explainer failed
    return "prepare_output"


workflow.add_conditional_edges(
    "generate_explainer", should_continue_after_explainer_gen
)


workflow.add_edge("prepare_output", END)
workflow.add_edge("prepare_error_output", END)

app = workflow.compile()


# --- Main Execution Logic ---
def process_single_image(image_path: str, compiled_graph):
    print(f"\\nProcessing image: {image_path}")
    initial_state = ExplainerAgentState(
        image_path=image_path,
        extracted_text=None,
        is_english=None,
        text_for_search_and_explainer=None,
        explainer_content=None,
        final_output_messages=[],
        error_message=None,
        was_translated=None,
    )
    try:
        final_state = compiled_graph.invoke(initial_state)
        if final_state and final_state.get("final_output_messages"):
            for msg in final_state["final_output_messages"]:
                print(msg)
        else:
            print(f"No final output messages for {image_path}. State: {final_state}")
    except Exception as e:
        print(f"An unhandled exception occurred while processing {image_path}: {e}")
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image Explainer Agent using LangGraph."
    )
    parser.add_argument(
        "path", help="Path to an image file or a directory containing images."
    )
    args = parser.parse_args()

    input_path = args.path

    if not os.path.exists(input_path):
        print(f"Error: Path does not exist: {input_path}")
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
        ):
            process_single_image(input_path, app)
        else:
            print(f"Error: File {input_path} is not a recognized image type.")
    elif os.path.isdir(input_path):
        print(f"Processing all images in directory: {input_path}")
        found_images = False
        for item in os.listdir(input_path):
            item_path = os.path.join(input_path, item)
            if os.path.isfile(item_path) and item.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
            ):
                process_single_image(item_path, app)
                found_images = True
        if not found_images:
            print(f"No image files found in directory: {input_path}")
    else:
        print(f"Error: Path {input_path} is not a valid file or directory.")
