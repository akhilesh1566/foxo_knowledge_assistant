# src/document_processor.py
# src/document_processor.py
import os
import fitz # PyMuPDF
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdfs(pdf_folder_path: str) -> List[Document]:
    """
    Loads all PDF files from a specified folder, extracts text and metadata.
    Improved error handling for individual PDF files.

    Args:
        pdf_folder_path (str): Path to the folder containing PDF files.

    Returns:
        List[Document]: A list of LangChain Document objects, where each
                        Document represents a page from a PDF.
    """
    documents = []
    total_pages_processed = 0
    files_processed_successfully = 0
    files_with_errors = 0

    if not os.path.isdir(pdf_folder_path):
        print(f"Error: Folder not found at {pdf_folder_path}")
        return documents

    print(f"Scanning PDF files in folder: {pdf_folder_path}")
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return documents

    print(f"Found {len(pdf_files)} PDF files to process.")

    for filename in pdf_files:
        file_path = os.path.join(pdf_folder_path, filename)
        pdf_document = None  # Initialize pdf_document to None
        try:
            print(f"\nProcessing file: {filename}...")
            pdf_document = fitz.open(file_path)
            pages_in_current_doc = 0
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text = page.get_text("text")
                if text.strip():
                    metadata = {
                        "source": filename,
                        "page": page_num + 1,
                        "file_path": file_path
                    }
                    documents.append(Document(page_content=text, metadata=metadata))
                    pages_in_current_doc += 1
            
            if pages_in_current_doc > 0:
                print(f"Successfully extracted {pages_in_current_doc} pages with text from {filename}.")
                total_pages_processed += pages_in_current_doc
                files_processed_successfully +=1
            else:
                print(f"No text content found or extracted from {filename}, though file was opened.")
                # Optionally, still count this as a "processed file" if it opened, 
                # or count it as an error/warning depending on expectation.
                # For now, we won't increment files_with_errors here if it just had no text.
                
        except Exception as e:
            print(f"ERROR processing file {filename}: {e}")
            files_with_errors += 1
        finally:
            if pdf_document: # Ensure document is closed only if it was successfully opened
                try:
                    pdf_document.close()
                except Exception as e_close:
                    # This might happen if it was already closed due to the initial error
                    print(f"Note: Exception while trying to explicitly close {filename}: {e_close}")
    
    print(f"\n--- PDF Loading Summary ---")
    print(f"Total PDF files found: {len(pdf_files)}")
    print(f"Files processed successfully (at least partially): {files_processed_successfully}")
    print(f"Files encountered errors during processing: {files_with_errors}")
    print(f"Total pages extracted with text content: {total_pages_processed}")
    
    if not documents:
        print(f"No PDF documents were successfully processed and converted to LangChain Documents.")
    
    return documents

def chunk_documents(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks.

    Args:
        documents (List[Document]): The list of Documents to chunk.
        chunk_size (int): The maximum size of each chunk (in characters).
        chunk_overlap (int): The overlap between consecutive chunks (in characters).

    Returns:
        List[Document]: A list of chunked LangChain Document objects.
    """
    if not documents:
        print("No documents provided for chunking.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document (pages) into {len(chunks)} chunks.")
    return chunks

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    print("Testing document_processor.py...")
    # Assume 'data' folder is in the parent directory of 'src'
    # For direct execution within src, adjust path or ensure execution from root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) 
    sample_pdf_folder = os.path.join(project_root, "data")

    if not os.path.exists(sample_pdf_folder):
        os.makedirs(sample_pdf_folder)
        print(f"Created sample data folder: {sample_pdf_folder}")
        print("Please add some PDF files to the 'data' folder to test.")
    else:
        raw_documents = load_pdfs(sample_pdf_folder)
        if raw_documents:
            print(f"\nLoaded {len(raw_documents)} pages in total.")
            for doc in raw_documents[:2]: # Print details of first two pages
                print(f"Content (first 50 chars): {doc.page_content[:50]}...")
                print(f"Metadata: {doc.metadata}")

            chunked_documents = chunk_documents(raw_documents)
            if chunked_documents:
                print(f"\nChunked into {len(chunked_documents)} smaller documents.")
                for chunk in chunked_documents[:2]: # Print details of first two chunks
                    print(f"Chunk content (first 50 chars): {chunk.page_content[:50]}...")
                    print(f"Chunk metadata: {chunk.metadata}")
                    # Note: RecursiveCharacterTextSplitter adds 'start_index' to metadata