# ingest.py
import os
from src.document_processor import load_pdfs, chunk_documents
from src.vector_store_manager import create_or_get_vector_store, get_embedding_function
from src.config import CHROMA_COLLECTION_NAME # Using the main collection name

# Define the path to your PDF data
# Assumes 'data' folder is in the same directory as ingest.py
DATA_PATH = "data"

def main():
    print("Starting document ingestion process...")

    # --- 1. Load PDF documents ---
    print(f"\nLoading PDFs from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"No files found in {DATA_PATH}. Please add PDF files to this directory.")
        print("Ingestion process cannot continue without data.")
        return
        
    raw_documents = load_pdfs(DATA_PATH)
    if not raw_documents:
        print("No documents were loaded. Exiting ingestion.")
        return
    print(f"Successfully loaded {len(raw_documents)} pages from PDF files.")

    # --- 2. Chunk documents ---
    print("\nChunking documents...")
    chunked_documents = chunk_documents(raw_documents) # Using default chunk_size and overlap
    if not chunked_documents:
        print("No chunks were created. Exiting ingestion.")
        return
    print(f"Successfully chunked documents into {len(chunked_documents)} pieces.")

    # --- 3. Initialize embedding function ---
    print("\nInitializing embedding function...")
    try:
        embedding_function = get_embedding_function()
        print("Embedding function initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize embedding function: {e}")
        return

    # --- 4. Create or update vector store ---
    print(f"\nCreating or updating vector store for collection: '{CHROMA_COLLECTION_NAME}'...")
    # Set recreate=True if you want to clear the collection and re-ingest every time.
    # For incremental additions, you'd load existing and add new (more complex logic).
    # For this POC, recreate=True ensures a fresh state on each run.
    try:
        vector_store = create_or_get_vector_store(
            documents=chunked_documents,
            embedding_function=embedding_function,
            collection_name=CHROMA_COLLECTION_NAME, 
            recreate=True # Set to True for POC to always rebuild
        )
        print(f"Vector store processing complete for collection '{CHROMA_COLLECTION_NAME}'.")
        
        # Verify by getting collection count directly from client
        import chromadb
        from src.config import CHROMA_PERSIST_DIRECTORY
        client = chromadb.PersistentClient(CHROMA_PERSIST_DIRECTORY)
        actual_collection= client.get_collection(CHROMA_COLLECTION_NAME)
        print(f"Verification: Collection '{CHROMA_COLLECTION_NAME}' now contains {actual_collection.count()} items.")

    except Exception as e:
        print(f"Failed to create or update vector store: {e}")
        return

    print("\nDocument ingestion process completed successfully!")

if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY is set
    from src.config import GOOGLE_API_KEY
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("ERROR: GOOGLE_API_KEY is not set or is still the placeholder.")
        print("Please create a .env file in the project root, add your key, for example:")
        print("GOOGLE_API_KEY=\"AIzaSy...\"")
        print("Ingestion cannot proceed without a valid API key.")
    else:
        main()