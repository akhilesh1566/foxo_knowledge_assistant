# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please set it up.")

# Define model names (these can be overridden in specific modules if needed)
# For embeddings, "models/embedding-001" is standard for google-generativeai SDK.
# For older models, it might be "text-embedding-004" but the newer is preferred.
GEMINI_EMBEDDING_MODEL = "models/embedding-001"

# For chat/generation, "gemini-pro" is a good default.
# "gemini-1.0-pro" is also common.
# "gemini-1.5-pro-latest" is newer but might have different availability/pricing.
GEMINI_CHAT_MODEL = "gemini-pro"

# ChromaDB settings
CHROMA_PERSIST_DIRECTORY = "vector_store"
CHROMA_COLLECTION_NAME = "foxo_docs_gemini"

# Print a confirmation that the key is loaded (optional, for debugging Phase 0)
# print(f"Config loaded. GOOGLE_API_KEY available: {bool(GOOGLE_API_KEY)}")