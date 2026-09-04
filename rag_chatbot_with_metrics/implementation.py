from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

API_KEY = os.getenv('GOOGLE_API_KEY')
MODEL = "gemini-3.1-flash-lite"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "vector_db")
RETRIEVAL_TOP_K = 5

if not API_KEY:
    raise ValueError("Gemini API Key not set")

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=API_KEY,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})


SYSTEM_PROMPT = """
    You are a knowledgeable, friendly assistant representing the company Insurellm.
    You are chatting with a user about Insurellm.
    If relevant, use the given context to answer any question.
    If you don't know the answer, say so.
    Context:
    {context}
    """


def _message_text(message):
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return _message_text(message.get("text", message.get("content", "")))
    if isinstance(message, list):
        return "".join(_message_text(part) for part in message)
    return str(message)


def get_combined_question(question, history):
    if isinstance(question, list):
        question = _message_text(question[-1])
    else:
        question = _message_text(question)
    if not history:
        return question
    combined_history = "\n".join(_message_text(msg) for msg in history)
    return f"{combined_history}\n{question}"

def answer_question(question, history):
    combined_ques = get_combined_question(question, history)
    docs = retriever.invoke(combined_ques)
    print(f"Retrieved {len(docs)} documents for question: {combined_ques}")
    for i, doc in enumerate(docs):
        print(f"Document {i+1}: Source: {doc.metadata['source']}, Content: {doc.page_content[:100]}...")
    print()
    context = "\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs

if __name__ == "__main__":
    # Example usage
    # print(retriever.invoke("What is Insurellm?"))
    question = [{"text": "What is Insurellm?"}]
    history = []
    answer, context = answer_question(question, history)
    print("Answer:", answer)
    print("Context:", context)