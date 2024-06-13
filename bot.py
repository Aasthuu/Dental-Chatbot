import os
from dotenv import load_dotenv
import google.generativeai as genai
import pickle
from typing import List, Optional
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import vertexai
from vertexai.generative_models import GenerativeModel, Part

class Content:
    def __init__(self, url, paragraphs):
        self.url = url
        self.paragraphs = paragraphs

def read_content():
    all_content = []
    with open('allContent.pkl', 'rb') as f:
        all_content = pickle.load(f)

    return all_content

def configure():
    # Ensure your OpenAI API key is set
    load_dotenv()

    API_KEY = os.getenv("API_KEY")


    genai.configure(api_key=API_KEY)

def embed_text(
    texts: List[str] = ["banana muffins? ", "banana bread? banana muffins?"],
    task: str = "RETRIEVAL_DOCUMENT",
    model_name: str = "text-embedding-004",
    dimensionality: Optional[int] = 256,
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings]
def main():
    #configure
    #configure()
    model = genai.GenerativeModel('gemini-1.5-flash')
    model_name = "gemini-1.5-flash"
    #response = model.generate_content("Write a short story about a cat who loves to eat pizza.")
    #print(response.text)

    project_id = "mimetic-fulcrum-407320"
    vertexai.init(project=project_id, location="us-central1")

    textEmbeddings = embed_text()
    print(textEmbeddings)


if __name__ == "__main__":
    main()