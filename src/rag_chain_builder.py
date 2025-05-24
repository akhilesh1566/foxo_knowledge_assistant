
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

from .config import (
    GOOGLE_API_KEY,
    GEMINI_CHAT_MODEL_FOR_RAG,
    GEMINI_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME
)
from .vector_store_manager import get_embedding_function # Re-use for consistency

# Global instance of vector store to avoid reloading multiple times if script/app runs long
_vector_store_instance = None

def get_vector_store_instance(
    embedding_fn = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY
):
    global _vector_store_instance
    if _vector_store_instance is None:
        if embedding_fn is None:
            embedding_fn = get_embedding_function()
        
        print(f"Loading vector store: collection='{collection_name}', persist_dir='{persist_directory}'")
        _vector_store_instance = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_fn,
            persist_directory=persist_directory
        )
        print(f"Vector store loaded. Found {_vector_store_instance._collection.count()} items.")
    return _vector_store_instance

def format_docs_with_sources(docs: list) -> str:
    """
    Formats retrieved documents into a string for the LLM context
    and includes source information for display.
    """
    if not docs:
        return "No context documents found."
    
    formatted_context_parts = []
    for i, doc in enumerate(docs):
        source_info = (
            f"Source {i+1} (File: {doc.metadata.get('source', 'N/A')}, "
            f"Page: {doc.metadata.get('page', 'N/A')})"
        )
        # Truncate page_content if too long for context, or ensure LLM can handle it
        content_snippet = doc.page_content[:1500] # Adjust snippet length as needed
        formatted_context_parts.append(f"{source_info}:\n{content_snippet}\n---\n")
    
    return "\n".join(formatted_context_parts)

def get_rag_chain(
    chat_model_name: str = GEMINI_CHAT_MODEL_FOR_RAG,
    k_retriever: int = 3 # Number of chunks to retrieve
):
    """
    Constructs and returns a RAG chain using Gemini models and ChromaDB.

    The returned chain expects a dictionary with a "question" key and
    will output a dictionary containing "question", "context_docs", and "answer".

    Args:
        chat_model_name (str): The name of the Gemini chat model to use.
        k_retriever (int): The number of top-k documents to retrieve from vector store.

    Returns:
        A LangChain Runnable.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not configured.")

    # Initialize LangChain components
    llm = ChatGoogleGenerativeAI(
        model=chat_model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1, # Lower for more factual RAG
        convert_system_message_to_human=True # Often helpful for Gemini
    )

    vector_store = get_vector_store_instance()
    retriever = vector_store.as_retriever(search_kwargs={"k": k_retriever})

    template = """
You are an AI assistant for answering questions based on the provided context.
Your task is to synthesize an answer from the retrieved document snippets.
If the context doesn't contain the answer, state that you cannot answer based on the provided information.
Do NOT use any external knowledge.
After providing the answer, list the sources you used from the context, including the Filename and Page number.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    prompt = ChatPromptTemplate.from_template(template)

    # --- RAG Chain Construction using LCEL ---
    
    # This chain is designed to:
    # 1. Take a "question" as input.
    # 2. Retrieve relevant documents ("context_docs") using the retriever.
    # 3. Pass the original "question" and the "context_docs" through.
    # 4. Format the "context_docs" into a string "context" for the LLM.
    # 5. Populate the prompt with "context" and "question".
    # 6. Send the populated prompt to the LLM.
    # 7. Parse the LLM's output string as the "answer".
    # The final output of the chain will be a dictionary:
    # {"question": original_question, "context_docs": retrieved_documents, "answer": llm_answer}

    rag_chain = (
        RunnablePassthrough.assign(
            context_docs=itemgetter("question") | retriever # Retrieve docs based on question
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs_with_sources(x["context_docs"]) # Format docs for LLM
        )
        | {
            "question": itemgetter("question"),
            "context_docs": itemgetter("context_docs"), # Pass through for final output
            "answer": prompt | llm | StrOutputParser() # Actual RAG call
          }
    )
    
    
    chain_components = {
        "context_docs": itemgetter("question") | retriever,
        "question": itemgetter("question"), # Pass the original question through
    }
    
    rag_chain_alternative = (
        RunnablePassthrough.assign(**chain_components) # Step 1: Retrieve and pass q
        | RunnablePassthrough.assign( # Step 2: Format context
            context=lambda x: format_docs_with_sources(x["context_docs"])
          )
        | RunnablePassthrough.assign( # Step 3: Generate answer, keep other keys
            answer= (
                prompt 
                | llm 
                | StrOutputParser()
            )
          )
        
        | (lambda x: {"question": x["question"], "context_docs": x["context_docs"], "answer": x["answer"]})
    )
    
    return rag_chain_alternative


if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    print("Testing RAG chain builder...")
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("GOOGLE_API_KEY not set. Please configure .env file.")
    else:
        try:
            # This will load the vector store the first time it's called
            rag_chain_instance = get_rag_chain() 
            print("RAG chain instance created successfully.")

            query = "What is the policy on acceptable use of IT resources?"

            print(f"\nInvoking RAG chain with question: '{query}'")
            result = rag_chain_instance.invoke({"question": query})

            print("\n--- RAG CHAIN OUTPUT ---")
            print(f"Question: {result.get('question')}")
            print(f"\nAnswer from LLM:\n{result.get('answer')}")
            
            print("\n--- Retrieved Context Documents ---")
            if result.get('context_docs'):
                for i, doc in enumerate(result.get('context_docs')):
                    print(f"\nSource Document {i+1}:")
                    print(f"  File: {doc.metadata.get('source', 'N/A')}")
                    print(f"  Page: {doc.metadata.get('page', 'N/A')}")
                    print(f"  Content Snippet (first 100 chars): {doc.page_content[:100]}...")
            else:
                print("No context documents were retrieved or passed through.")

        except Exception as e:
            print(f"Error during RAG chain test: {e}")
            import traceback
            traceback.print_exc()
