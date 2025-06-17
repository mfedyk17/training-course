__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DataFrameLoader
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import Chroma


def create_and_store_embeddings(documents_to_embed: list, api_key: str) -> Chroma:
    """
    Initializes an embedding model and creates a vector store from documents.

    This function takes a list of prepared text documents and converts them into
    numerical vectors (embeddings) using an AI model. It then stores these
    embeddings in a searchable Chroma vector database.

    Args:
        documents_to_embed: A list of chunked LangChain Document objects.
        api_key: The API key required to use the OpenAI embedding service.

    Returns:
        A Chroma vector store object that contains the documents and their embeddings.
    """

    # 1. Initialize the Embedding Model
    # An "embedding model" is a specialized AI that reads text and converts it into
    # a fixed-size list of numbers called a "vector". This vector captures the
    # semantic meaning of the text, allowing us to compare texts mathematically.
    # Here, we're using OpenAI's "text-embedding-3-small" model, which offers a great
    # balance of performance and cost.
    embeddings_model = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

    # 2. Create the Vector Store
    # A "vector store" or "vector database" is a special kind of database designed to
    # efficiently store and search for vectors.
    # The `Chroma.from_documents` function is a powerful helper that does two things at once:
    #   a. It uses the `embeddings_model` to create a vector for every document you provide.
    #   b. It stores both the original document content and its new vector in the Chroma database.
    # This process of creating vectors and storing them is often called "ingestion".
    # --- NOTE: Converting the documents into vectors takes time, for this example, expect about 30 seconds. ---
    vectorstore = Chroma.from_documents(
            documents = documents_to_embed,  # The list of prepared Document objects.
            # collection_metadata = document_metadata,
            embedding = embeddings_model,    # The model to use for the embedding process.
            # persist_directory="./chroma_db",   # Optional: You can uncomment this line to save the
            # client_settings=Settings(chroma_db_impl="duckdb+parquet")                                  # database to a folder on your computer. This allows
                                               # you to load it again later without re-running the process if it is time consuming
        )

    # Return the fully populated vector store object, which is now ready to be used for searches.
    return vectorstore


dataset = load_dataset("dgervais/patient-doctor", split="train")
notes = dataset.to_pandas()
documents = [
    Document(
        page_content=row["text_lemmatized"],
        metadata={
            "doctor_name": row["doctor_name"]
        }
    )
    for _, row in notes.iterrows()
]

# --- Helper functions for RAG ---

# Define a helper function to format the retrieved documents into a single string.
# The retriever will return a list of Document objects. This function takes that list,
# extracts the 'page_content' (the actual text) from each, and joins them together
# with two newline characters in between. This creates a clean, readable context block.
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] Doctor: {d.metadata.get('doctor_name', 'Unknown')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Streamlit AI Chat Assistant")
st.markdown("""
Welcome! Ask any question to the AI assistant. This application uses OpenAI's `gpt-4o-mini` model.
Enter your OpenAI API Key in the sidebar to begin.
""")

# --- API Key Handling ---
openai_api_key = None

# Attempt to get API key from st.secrets (for deployed apps)
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    if openai_api_key:
        st.sidebar.success("API key loaded from st.secrets!")
    else: # Handle case where secret exists but is empty
        st.sidebar.warning("OpenAI API Key found in st.secrets but it's empty. Please provide a valid key.")
except (KeyError, FileNotFoundError): # FileNotFoundError for local st.secrets.toml if used
    st.sidebar.info("OpenAI API Key not found in st.secrets. Please enter it below for this session.")

# Fallback to user input if not found in secrets or if secret was empty
if not openai_api_key:
    openai_api_key_input = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        key="api_key_input_sidebar",
        help="Your API key is used only for this session and not stored."
    )
    if openai_api_key_input:
        openai_api_key = openai_api_key_input

if not openai_api_key:
    st.warning("Please provide your OpenAI API Key in the sidebar to use the chat.")
    st.stop() # Stop execution if no API key is available

@st.cache_resource
def load_vectorstore(api_key: str):
    return create_and_store_embeddings(documents, api_key)

vector_store = load_vectorstore(openai_api_key)
get_query = RunnableLambda(lambda d: d["user_input"])

# --- LangChain Setup (Cached for efficiency) ---
@st.cache_resource # Caches the LLM and prompt template
def get_langchain_components(_api_key_for_cache): # Parameter ensures cache reacts to API key changes if necessary
    """Initializes and returns the LangChain LLM and prompt template."""
    llm = ChatOpenAI(temperature=0.7, api_key=_api_key_for_cache, model_name="gpt-4o-mini")
    # llm = ChatOpenAI(openai_api_key=_api_key_for_cache, model_name="gpt-4o-mini")
    
    # prompt_template_str = """
    # You are a knowledgeable and friendly AI assistant.
    # Your goal is to provide clear, concise, and helpful answers to the user's questions.
    # If you don't know the answer to a specific question, it's better to say so rather than inventing one.

    # User Question: {user_input}

    # AI Response:
    # """
    prompt_template_str = """
    You are a helpful assistant.
    Analyze the context and find right doctor based on symptom the user input.

    If you don't know the answer to a specific question, it's better to say so rather than inventing one.

    Context:
    {context}

    Question: {question}

    AI Response:"""

    
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    return llm, prompt

try:
    llm, prompt_template = get_langchain_components(openai_api_key)
except Exception as e:
    st.error(f"Failed to initialize AI components. Error: {e}. Check your API key and model access.")
    st.stop()

# --- Initialize session state for storing chat messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display existing chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and AI Response Logic ---
if user_query := st.chat_input("What would you like to ask?"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Get and display AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # For "Thinking..." message and then the actual response
        with st.spinner("AI is thinking..."):
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                rag_chain = (
        {"context": get_query | retriever | format_docs, "question": get_query | RunnablePassthrough()}
        | prompt_template  
        | llm           
        | StrOutputParser() 
    )
                ai_response_message = rag_chain.invoke({"user_input": user_query})
                ai_response_content = ai_response_message
                
                message_placeholder.markdown(ai_response_content)
                # Add AI response to session state
                st.session_state.messages.append({"role": "assistant", "content": ai_response_content})

            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Sidebar Options ---
with st.sidebar:
    st.divider()
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun() # Rerun to update the UI immediately

    st.markdown("---")
    st.subheader("About")
    st.info(
        "This is a Streamlit application about doctor recommendation system based on symptom decription "
        "using LangChain and OpenAI's gpt-4o-mini model."
    )