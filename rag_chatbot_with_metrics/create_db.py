import os
import glob
import tiktoken
import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_docling.loader import DoclingLoader, ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge-base")
DB_DIR = os.path.join(BASE_DIR, "vector_db")


def get_chars_in_kb():
    knowledge_base_path = os.path.join(KB_DIR, "**", "*.md")
    files = glob.glob(knowledge_base_path, recursive=True)
    print(f"Found {len(files)} files in the knowledge base")

    entire_knowledge_base = ""

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            entire_knowledge_base += f.read()
            entire_knowledge_base += "\n\n"

    print(f"Total characters in knowledge base: {len(entire_knowledge_base):,}")

    return entire_knowledge_base

def get_tokens_in_kb(model):
    entire_knowledge_base = get_chars_in_kb()
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(entire_knowledge_base)
    token_count = len(tokens)
    print(f"Total tokens for {model}: {token_count:,}")

def create_db():
    load_dotenv(override=True)
    gemini_api_key = os.getenv('GOOGLE_API_KEY')
    if gemini_api_key:
        print(f"Gemini API Key exists and begins {gemini_api_key[:8]}")
    else:
        print("Gemini API Key not set")

    base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'
    MODEL = "gemini-3.1-flash-lite"

    # Load in everything in the knowledgebase using LangChain's loaders

    folders = glob.glob(os.path.join(KB_DIR, "*"))
    documents = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue

        doc_type = os.path.basename(folder)
        file_paths = [
            path for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
            if os.path.isfile(path)
        ]

        for file_path in file_paths:
            loader = DoclingLoader(file_path, export_type=ExportType.MARKDOWN)
            folder_docs = loader.load()
            for doc in folder_docs:
                clean_metadata = {
                    "source": os.path.basename(doc.metadata.get("source", file_path)),
                    "doc_type": doc_type,
                }
                doc.metadata = clean_metadata
                documents.append(doc)

    print(f"Loaded {len(documents)} documents")
    # print(documents[1])

    # Divide into chunks using the MarkdownTextSplitter
    text_splitter = MarkdownTextSplitter(chunk_size=850, chunk_overlap=250)
    chunks = text_splitter.split_documents(documents)

    print(f"Divided into {len(chunks)} chunks")
    # print(f"First chunk:\n\n{chunks[0]}")

    #Create embeddings and store in Chroma vector database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    print(f"Persisting vector DB to: {DB_DIR}")
    if os.path.exists(DB_DIR):
        Chroma(persist_directory=DB_DIR, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")



if __name__ == "__main__":
    create_db()