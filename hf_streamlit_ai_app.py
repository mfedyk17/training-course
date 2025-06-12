# Block 2: Create the Streamlit Application File (app.py)
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from datasets import load_dataset

load_dotenv()
api_key = os.environ['OPENAI_API_KEY']

def create_and_store_embeddings(documents_to_embed: list, document_metadata: list, api_key: str) -> Chroma:
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
            collection_metadata = document_metadata,
            embedding = embeddings_model,    # The model to use for the embedding process.
            # persist_directory="./chroma_db"  # Optional: You can uncomment this line to save the
                                               # database to a folder on your computer. This allows
                                               # you to load it again later without re-running the process if it is time consuming
        )

    # Return the fully populated vector store object, which is now ready to be used for searches.
    return vectorstore

dataset = load_dataset("dgervais/patient-doctor")
notes = dataset["train"].to_pandas()

loader = DataFrameLoader(notes, page_content_column="text_lemmatized")
documents = loader.load()
vector_store = create_and_store_embeddings(documents, api_key)
# ------

# --- Helper functions for RAG ---

# Define a helper function to format the retrieved documents into a single string.
# The retriever will return a list of Document objects. This function takes that list,
# extracts the 'page_content' (the actual text) from each, and joins them together
# with two newline characters in between. This creates a clean, readable context block.
def format_docs(docs):
    return "\n\n".join(doc.text_lemmatized for doc in docs)

# This is the main function that orchestrates the entire RAG process.
def generate_rag_response(input_text: str, _vectorstore, _openai_api_key: str) -> str:
    """
    Generates a response using the RAG pattern.

    Args:
        input_text: The user's question.
        _vectorstore: The Chroma vector store containing the document embeddings.
        _openai_api_key: The API key for the OpenAI service.

    Returns:
        The AI-generated response as a string.
    """
    # A quick safety check to ensure the vector store has been created before we try to use it.
    if _vectorstore is None:
        return "Error: Vector store not available."

    # Initialize the Language Model (LLM) we'll use to generate the final answer.
    # 'temperature=0.7' controls the creativity of the model; higher values mean more creative,
    # lower values mean more deterministic and factual.
    llm = ChatOpenAI(temperature=0.7, api_key=_openai_api_key)

    # Create a "retriever" from our vector store. A retriever is an object that can
    # "retrieve" documents from a database based on a query.
    # 'search_kwargs={"k": 3}' tells the retriever to find the top 3 most relevant
    # document chunks for any given question.
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define the prompt template. This is the heart of our instruction to the AI.
    # It sets the persona (a literary assistant), gives clear instructions on how to behave,
    # and defines where the retrieved 'context' and user's 'question' will be inserted.
    prompt_template = """You are a helpful assistant.
    Analyze the context below to answer the user's question.

    # # # TODO: # # # - MODIFY THIS PROMPT TO IMRPOVE YOUR RESPONSE! # # #

    Context:
    {context}

    Question: {question}

    Answer:"""

    # Create a ChatPromptTemplate object from the string template defined above.
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Now, we create the full RAG chain using LangChain Expression Language (LCEL).
    # The '|' (pipe) operator connects the different components in a sequence.
    rag_chain = (
        # This first step runs in parallel. It creates a dictionary containing the 'context' and 'question'.
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        # 1a. "context": The user's input goes to the 'retriever', which finds relevant docs.
        #     The list of docs is then passed to our 'format_docs' function to create the context string.
        # 1b. "question": 'RunnablePassthrough' simply takes the user's original input (the question)
        #     and passes it through unchanged.

        | prompt          # 2. The dictionary from the previous step is "piped" into the 'prompt' template.
                          #    This fills in the {context} and {question} placeholders.

        | llm             # 3. The fully-formatted prompt is sent to the language model ('llm') to generate an answer.

        | StrOutputParser() # 4. The model's output (a chat message object) is converted into a simple string.
    )

    # Use a try-except block for robust error handling, especially for API calls.
    try:
        # ".invoke()" is the command that runs the entire chain with the user's question.
        response = rag_chain.invoke(input_text)
        return response
    except Exception as e:
        # If anything goes wrong during the chain execution, print the error.
        print(f"Error generating response: {e}")
        return "Sorry, an error occurred while generating the response."
# ------

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

# --- LangChain Setup (Cached for efficiency) ---
@st.cache_resource # Caches the LLM and prompt template
def get_langchain_components(_api_key_for_cache): # Parameter ensures cache reacts to API key changes if necessary
    """Initializes and returns the LangChain LLM and prompt template."""
    llm = ChatOpenAI(openai_api_key=_api_key_for_cache, model_name="gpt-4o-mini")
    
    prompt_template_str = """
    You are a knowledgeable and friendly AI assistant.
    Your goal is to provide clear, concise, and helpful answers to the user's questions.
    If you don't know the answer to a specific question, it's better to say so rather than inventing one.

    User Question: {user_input}

    AI Response:
    """
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
                chain = prompt_template | llm
                ai_response_message = chain.invoke({"user_input": user_query})
                ai_response_content = ai_response_message.content
                
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
        "This is a Streamlit application demonstrating an AI chat interface "
        "using LangChain and OpenAI's gpt-4o-mini model."
    )


# --- Block 3: Run the Streamlit Application ---

# To start your streamlit app, run the following command in your terminal:

# streamlit run hf_streamlit_ai_app.py
# Make sure you have the required packages installed:
# pip install streamlit langchain openai langchain_openai -q