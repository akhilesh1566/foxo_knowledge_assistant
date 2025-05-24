# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Add this

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found. RAG tool might fail if it uses Gemini LLM internally.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file. Needed for AutoGen Assistant.")

# --- Model Names ---
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_CHAT_MODEL_FOR_RAG = "gemini-2.0-flash-001" # Used *inside* the RAG tool

# OpenAI Model for AutoGen Assistant Agent
OPENAI_MODEL_FOR_ASSISTANT = "gpt-3.5-turbo-0125" # or "gpt-4-turbo-preview"

# --- ChromaDB Settings ---
CHROMA_PERSIST_DIRECTORY = "vector_store"
CHROMA_COLLECTION_NAME = "foxo_docs_gemini" # Sticking with Gemini in name for embeddings

# --- AutoGen Configuration for Assistant (OpenAI) ---
AUTOGEN_OPENAI_CONFIG_LIST = [
    {
        "model": OPENAI_MODEL_FOR_ASSISTANT,
        "api_key": OPENAI_API_KEY,
    }
]