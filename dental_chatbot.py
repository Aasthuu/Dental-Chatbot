from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import pickle

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from bot import Content
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_google_community import GoogleTranslateTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_core import prompts
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from googlesearch import search
from langchain.agents import Tool, initialize_agent

import streamlit as st

import vertexai
from vertexai.preview import reasoning_engines



model_kwargs = {
    # temperature (float): The sampling temperature controls the degree of
    # randomness in token selection.
    "temperature": 0.28,
    # max_output_tokens (int): The token limit determines the maximum amount of
    # text output from one prompt.
    "max_output_tokens": 1000,
    # top_p (float): Tokens are selected from most probable to least until
    # the sum of their probabilities equals the top-p value.
    "top_p": 0.95,
    # top_k (int): The next token is selected from among the top-k most
    # probable tokens.
    "top_k": 40
}

# Load environment variables from the .env file
def configure():
    load_dotenv()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Get the API key from the environment variable
    api_key = os.getenv('API_KEY')

    if not api_key:
        raise ValueError("API key not found. Please set the MISTRAL_API_KEY environment variable.")

    os.environ['GOOGLE_API_KEY'] = api_key
    project_id = "mimetic-fulcrum-407320"
    vertexai.init(project=project_id, location="us-central1")

def read_content():
    all_content = []
    with open('allContent.pkl', 'rb') as f:
        all_content = pickle.load(f)

    return all_content

"""
Just a test file
"""
def attention_pdfloader():
    loader = PyPDFLoader('attention.pdf')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    return documents[:5]

def get_dental_documents():
    allDocuments=[]
    files = read_content()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10 )
    for file in files:
        for para in file.paragraphs:
            if para.strip(): #process if para is not empty
                split_texts = text_splitter.split_text(para)
                for text_chunk in split_texts:
                    document = Document(page_content=text_chunk, metadata={"url": file.url})
                    allDocuments.append(document)
    return allDocuments

def similarity_search(db, query):

    print("Ask question...")
    query = "How to clean teeth"
    retireved_results = db.similarity_search(query)
    print("Printing result")
    print(retireved_results[0].page_content)


def load_db_with_embeddings(documents):
    print("Applying embeddings Google...")
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Store to FAISS applying gemini embeddings...")

    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        gemini_embeddings, store, namespace=gemini_embeddings.model
    )
    list(store.yield_keys())

    db = FAISS.from_documents(documents, cached_embedder)
    return db

template = """Question: {question}

Answer: Let's think step by step."""

def main():
    configure()
    documents=get_dental_documents()
    db = load_db_with_embeddings(documents)
    print("Total Indexes created?", db.index.ntotal)
    model = VertexAI(model_name="gemini-1.0-pro-001")
    retriever = db.as_retriever()

    user_prompt_model = """Given the user query {query} , present your answer in 3 sentence 
    and make it as clear and concise"""

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question, Use three sentences maximum and keep the "
        "answer concise. If you don't know the answer, just reply 'NoIdea' "
    
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    question = "How to clean toilet"
    result = rag_chain.invoke({"input": question})
    answer = result['answer']
    if(answer=='NoIdea'):
        ai_prompt = PromptTemplate.from_template(user_prompt_model)
        chain = ai_prompt | model
        answer = chain.invoke({"query": question})

        tool = Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=search,
        )
        tool.run("Obama's first name?")

        agent = initialize_agent(
            tools,
            self.llm.model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent

    print(answer)







if __name__ == "__main__":
    main()