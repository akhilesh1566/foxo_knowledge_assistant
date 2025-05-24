import os
import fitz  # PyMuPDF for PDFs
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import traceback # For more detailed error logging

def load_supported_documents(folder_path: str) -> List[Document]:
    """
    Loads all supported document files (.pdf, .txt, .md) from a specified folder,
    extracts their content, and associates metadata.

    For PDFs, each page becomes a Document.
    For TXT and MD files, the entire file content becomes a single Document.
    MD files are read as plain text for simplicity in this version.

    Args:
        folder_path (str): Path to the folder containing document files.

    Returns:
        List[Document]: A list of LangChain Document objects.
    """
    all_langchain_documents: List[Document] = []
    files_processed_successfully = 0
    files_with_errors = 0
    # total_pages_or_files_processed is a bit ambiguous, len(all_langchain_documents) is better
    
    if not os.path.isdir(folder_path):
        print(f"ERROR: Document folder not found at '{folder_path}'")
        return all_langchain_documents

    print(f"INFO: Scanning document files in folder: '{folder_path}'")
    
    supported_extensions = (".pdf", ".txt", ".md")
    filenames_to_process = [
        f for f in os.listdir(folder_path) 
        if f.lower().endswith(supported_extensions)
    ]
    
    if not filenames_to_process:
        print(f"INFO: No supported document files found ({', '.join(supported_extensions)}) in '{folder_path}'.")
        return all_langchain_documents

    print(f"INFO: Found {len(filenames_to_process)} supported files to process.")

    for filename in filenames_to_process:
        file_path = os.path.join(folder_path, filename)
        file_extension = os.path.splitext(filename.lower())[1] # Gets ".pdf", ".txt", etc.
        
        try:
            print(f"\nINFO: Processing file: '{filename}' (type: {file_extension})...")
            documents_from_current_file: List[Document] = []

            if file_extension == ".pdf":
                pdf_document_handle = None
                try:
                    pdf_document_handle = fitz.open(file_path)
                    for page_num in range(len(pdf_document_handle)):
                        page = pdf_document_handle.load_page(page_num)
                        text = page.get_text("text")
                        if text.strip():
                            metadata = {
                                "source": filename,
                                "page": page_num + 1,
                                "file_path": file_path,
                                "type": "pdf"
                            }
                            documents_from_current_file.append(Document(page_content=text, metadata=metadata))
                    if documents_from_current_file:
                        print(f"INFO: Successfully extracted {len(documents_from_current_file)} pages with text from PDF '{filename}'.")
                except Exception as e_pdf:
                    print(f"ERROR processing PDF file '{filename}': {e_pdf}")
                    traceback.print_exc() # Log full traceback for PDF errors
                    files_with_errors += 1
                    continue # Skip to next file if PDF processing fails fundamentally
                finally:
                    if pdf_document_handle:
                        try:
                            pdf_document_handle.close()
                        except Exception as e_close:
                            print(f"WARN: Exception while trying to explicitly close PDF '{filename}': {e_close}")
            
            elif file_extension in [".txt", ".md"]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if content.strip():
                        metadata = {
                            "source": filename,
                            "file_path": file_path,
                            "type": file_extension.strip('.') # "txt" or "md"
                            # No 'page' number for these file types
                        }
                        documents_from_current_file.append(Document(page_content=content, metadata=metadata))
                        print(f"INFO: Successfully loaded content from {file_extension.upper()} file '{filename}'.")
                    else:
                        print(f"INFO: {file_extension.upper()} file '{filename}' is empty or contains only whitespace.")
                except Exception as e_text_md:
                    print(f"ERROR reading {file_extension.upper()} file '{filename}': {e_text_md}")
                    traceback.print_exc()
                    files_with_errors += 1
                    continue # Skip to next file

            else: # Should not happen due to initial filtering
                print(f"WARN: Skipped unsupported file type: '{filename}'")
                continue

            if documents_from_current_file:
                all_langchain_documents.extend(documents_from_current_file)
                files_processed_successfully += 1
            # If documents_from_current_file is empty (e.g. empty PDF or TXT/MD), it's not counted as success here.
            # files_with_errors is incremented only on exceptions.

        except Exception as e_outer: # Catch-all for unexpected issues with a file
            print(f"CRITICAL ERROR processing file '{filename}': {e_outer}")
            traceback.print_exc()
            files_with_errors += 1
            
    print(f"\n--- Document Loading Summary ---")
    print(f"Total supported files found: {len(filenames_to_process)}")
    print(f"Files processed that yielded at least one Document object: {files_processed_successfully}")
    print(f"Files that encountered errors during processing: {files_with_errors}")
    print(f"Total LangChain Document objects created: {len(all_langchain_documents)}")
    
    if not all_langchain_documents:
        print(f"INFO: No documents were successfully processed and loaded from '{folder_path}'.")
    
    return all_langchain_documents


def chunk_documents(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks.
    Ensures each chunk retains metadata from its parent document.

    Args:
        documents (List[Document]): The list of Documents to chunk.
        chunk_size (int): The maximum size of each chunk (in characters).
        chunk_overlap (int): The overlap between consecutive chunks (in characters).

    Returns:
        List[Document]: A list of chunked LangChain Document objects.
    """
    if not documents:
        print("INFO: No documents provided for chunking.")
        return []

    # Ensure basic metadata like 'source' is present before splitting
    # This loop is good practice, although our loaders above should set 'source'.
    for doc in documents:
        if "source" not in doc.metadata:
            # Fallback if source wasn't set by the loader appropriately
            doc.metadata["source"] = doc.metadata.get("file_path", "Unknown_Source_File")
            print(f"WARN: Document from '{doc.metadata['source']}' was missing 'source' in metadata, added fallback.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Helpful for potential future advanced retrieval needs
    )
    
    chunks = text_splitter.split_documents(documents)
    # RecursiveCharacterTextSplitter automatically carries over metadata from parent docs to chunks.
    print(f"INFO: Split {len(documents)} Document object(s) into {len(chunks)} chunk(s).")
    return chunks


if __name__ == '__main__':
    print("--- Testing document_processor.py ---")
    
    current_script_path = os.path.abspath(__file__)
    src_directory = os.path.dirname(current_script_path)
    project_root = os.path.dirname(src_directory) 
    sample_data_folder = os.path.join(project_root, "data")

    print(f"Attempting to load documents from: {sample_data_folder}")

    if not os.path.exists(sample_data_folder):
        try:
            os.makedirs(sample_data_folder)
            print(f"INFO: Created sample data folder: '{sample_data_folder}'")
            print("INFO: Please add some .pdf, .txt, or .md files to the 'data' folder to test.")
        except OSError as e:
            print(f"ERROR: Could not create data folder '{sample_data_folder}': {e}")
    else:
        raw_documents = load_supported_documents(sample_data_folder)
        if raw_documents:
            print(f"\n--- Sample of Loaded Documents (Max 2) ---")
            for i, doc in enumerate(raw_documents[:min(2, len(raw_documents))]): 
                print(f"\nDocument {i+1}:")
                # CORRECTED LINE:
                content_snippet = doc.page_content[:70].strip().replace('\n', ' ')
                print(f"  Content (first 70 chars): '{content_snippet}...'")
                print(f"  Metadata: {doc.metadata}")

            chunked_documents = chunk_documents(raw_documents)
            if chunked_documents:
                print(f"\n--- Sample of Chunked Documents (Max 2) ---")
                for i, chunk in enumerate(chunked_documents[:min(2, len(chunked_documents))]): 
                    print(f"\nChunk {i+1}:")
                    # CORRECTED LINE:
                    chunk_snippet = chunk.page_content[:70].strip().replace('\n', ' ')
                    print(f"  Content (first 70 chars): '{chunk_snippet}...'")
                    print(f"  Metadata: {chunk.metadata}")
        else:
            print("INFO: No documents were loaded in the test run.")
    print("\n--- End of document_processor.py test ---")