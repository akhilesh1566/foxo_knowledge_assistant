# ingest.py
import os
# Updated import to use the new function name
from src.document_processor import load_supported_documents, chunk_documents
from src.vector_store_manager import create_or_get_vector_store, get_embedding_function
from src.config import CHROMA_COLLECTION_NAME 

DATA_PATH = "data"

def main():
    print("Starting document ingestion process (supports .pdf, .txt, .md)...") # Updated message

    # --- 1. Load supported documents ---
    print(f"\nINFO: Loading documents from: '{DATA_PATH}'") # Changed to INFO and path in quotes
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"ERROR: No files found in '{DATA_PATH}'. Please add PDF, TXT, or MD files to this directory.") # Updated message
        print("INFO: Ingestion process cannot continue without data.")
        return
        
    raw_documents = load_supported_documents(DATA_PATH) # <-- UPDATED FUNCTION CALL
    if not raw_documents:
        print("ERROR: No documents were loaded. Exiting ingestion.") # Changed to ERROR
        return
    # Updated print statement to be more generic as raw_documents contains more than just pages
    print(f"INFO: Successfully created {len(raw_documents)} LangChain Document object(s) from files.")

    # --- 2. Chunk documents ---
    print("\nINFO: Chunking documents...")
    chunked_documents = chunk_documents(raw_documents) # Using default chunk_size and overlap
    if not chunked_documents:
        print("ERROR: No chunks were created. Exiting ingestion.") # Changed to ERROR
        return
    print(f"INFO: Successfully chunked documents into {len(chunked_documents)} pieces.")

    # --- 3. Initialize embedding function ---
    print("\nINFO: Initializing embedding function...")
    try:
        embedding_function = get_embedding_function()
        print("INFO: Embedding function initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize embedding function: {e}")
        return

    # --- 4. Create or update vector store ---
    print(f"\nINFO: Creating or updating vector store for collection: '{CHROMA_COLLECTION_NAME}'...")
    try:
        # recreate=True is good for this POC to ensure a fresh state on each run.
        # For production, you might want more sophisticated update logic.
        vector_store = create_or_get_vector_store(
            documents=chunked_documents,
            embedding_function=embedding_function,
            collection_name=CHROMA_COLLECTION_NAME, 
            recreate=True 
        )
        print(f"INFO: Vector store processing complete for collection '{CHROMA_COLLECTION_NAME}'.")
        
        # Verify by getting collection count directly from client
        import chromadb
        from src.config import CHROMA_PERSIST_DIRECTORY # Ensure this is correctly defined in your config
        client = chromadb.PersistentClient(CHROMA_PERSIST_DIRECTORY)
        actual_collection = client.get_collection(CHROMA_COLLECTION_NAME)
        # Added a check if collection exists before trying to count, though create_or_get_vector_store should handle it.
        if actual_collection:
            print(f"INFO: Verification: Collection '{CHROMA_COLLECTION_NAME}' now contains {actual_collection.count()} items.")
        else:
            print(f"WARN: Collection '{CHROMA_COLLECTION_NAME}' not found after attempting creation/update.")


    except Exception as e:
        print(f"ERROR: Failed to create or update vector store: {e}")
        import traceback # Add traceback for vector store errors
        traceback.print_exc()
        return

    print("\nINFO: Document ingestion process completed successfully!")

if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY is set for embeddings
    from src.config import GOOGLE_API_KEY
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("ERROR: GOOGLE_API_KEY is not set or is still the placeholder.")
        print("       Please create a .env file in the project root, add your key, for example:")
        print("       GOOGLE_API_KEY=\"AIzaSy...\"")
        print("       Ingestion cannot proceed without a valid API key for embeddings.")
    else:
        main()