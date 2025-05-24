# src/vector_store_manager.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb # For persistent client
from typing import List
from langchain_core.documents import Document

from .config import (
    GOOGLE_API_KEY, 
    GEMINI_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME
)

def get_embedding_function():
    """
    Initializes and returns the Google Generative AI embedding function.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not configured. Please ensure it's in .env and loaded.")
    
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
        # task_type="retrieval_document" # Optional: specify task type
    )

def create_or_get_vector_store(
    documents: List[Document] = None, 
    embedding_function=None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    recreate: bool = False # Flag to force recreation
) -> Chroma:
    """
    Creates a new Chroma vector store from documents or loads an existing one.

    Args:
        documents (List[Document], optional): List of documents to add. 
                                              Required if creating a new store.
        embedding_function: The embedding function to use.
        collection_name (str): Name of the collection in ChromaDB.
        persist_directory (str): Directory to persist ChromaDB data.
        recreate (bool): If True, will try to delete existing collection and recreate.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    if embedding_function is None:
        embedding_function = get_embedding_function()

    if recreate:
        print(f"Recreate flag is True. Attempting to delete collection '{collection_name}' if it exists.")
        try:
            persistent_client = chromadb.PersistentClient(path=persist_directory)
            # Check if collection exists before trying to delete
            existing_collections = [col.name for col in persistent_client.list_collections()]
            if collection_name in existing_collections:
                persistent_client.delete_collection(name=collection_name)
                print(f"Successfully deleted existing collection: {collection_name}")
            else:
                print(f"Collection '{collection_name}' not found, no need to delete.")
        except Exception as e:
            print(f"Error during collection deletion (continuing to create): {e}")


    if documents:
        print(f"Creating new vector store with {len(documents)} documents in collection '{collection_name}'.")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        print(f"Vector store created and persisted to {persist_directory}.")
    else:
        print(f"Loading existing vector store from collection '{collection_name}' in {persist_directory}.")
        # This assumes the collection and persist directory exist
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
        )
        print("Existing vector store loaded.")
    return vector_store

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    print("Testing vector_store_manager.py...")
    try:
        test_embedding_function = get_embedding_function()
        print(f"Successfully initialized embedding function for model: {GEMINI_EMBEDDING_MODEL}")

        # Create dummy documents for testing
        test_docs = [
            Document(page_content="This is a test document for Chroma.", metadata={"source": "test.txt", "page": 1}),
            Document(page_content="Another test document here.", metadata={"source": "test.txt", "page": 2})
        ]
        
        print("\nAttempting to create a new vector store (test_collection)...")
        # Use a different collection name for direct testing to avoid overwriting main one
        test_collection_name = "test_foxo_collection"
        vs = create_or_get_vector_store(
            documents=test_docs, 
            embedding_function=test_embedding_function,
            collection_name=test_collection_name,
            recreate=True # Ensure clean state for test
        )
        
        # Verify count in the created collection
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        collection = chroma_client.get_collection(name=test_collection_name, embedding_function=test_embedding_function.embed_documents) # embed_documents needed by raw client
        count = collection.count()
        print(f"Number of items in '{test_collection_name}': {count}")
        assert count == len(test_docs), "Item count in test collection mismatch!"

        print("\nAttempting to load the existing vector store (test_collection)...")
        vs_loaded = create_or_get_vector_store(
            embedding_function=test_embedding_function, # No documents, so it should load
            collection_name=test_collection_name
        )
        
        # Test similarity search
        if vs_loaded:
            query = "test document"
            print(f"\nPerforming similarity search for: '{query}'")
            results = vs_loaded.similarity_search(query, k=1)
            if results:
                print(f"Found {len(results)} result(s):")
                for doc in results:
                    print(f" - Content: {doc.page_content[:50]}...")
                    print(f"   Metadata: {doc.metadata}")
            else:
                print("No results found for similarity search.")
        
        print("\nVector store manager test completed successfully.")
        
        # Optional: cleanup test collection
        # persistent_client.delete_collection(name=test_collection_name)
        # print(f"Cleaned up test collection: {test_collection_name}")

    except Exception as e:
        print(f"Error during vector_store_manager.py test: {e}")