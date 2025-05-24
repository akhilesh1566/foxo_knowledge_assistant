# test_rag.py
import os
from src.rag_chain_builder import get_rag_chain
from src.config import GOOGLE_API_KEY

def run_rag_test():
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("ERROR: GOOGLE_API_KEY is not set or is still the placeholder.")
        print("Please ensure your .env file is correctly set up.")
        return

    print("Initializing RAG chain...")
    try:
        rag_chain = get_rag_chain() # Uses default models from config
        print("RAG chain initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize RAG chain: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Ask Questions ---
    # Try questions relevant to the content of your PDFs.
    # Examples (you'll need to adjust these based on your actual PDF content):
    # From "Global-AI-Policy-V1.pdf" (if it discusses ethical AI, data usage, etc.)
    # questions = [
    # "What are the ethical considerations for using AI according to the policy?",
    # "How should employees handle data when developing AI systems?"
    # ]

    # From "ITPOLICYFINAL.pdf" (if it discusses password rules, internet usage, etc.)
    questions = [
        "What is the company's policy on acceptable use of IT resources?",
        "How often should passwords be changed?", # If your IT policy mentions this
        "Are personal devices allowed on the company network?" # If your IT policy mentions this
    ]
    
    # A question likely not in the documents, to test the "I don't know" response
    questions.append("What is the capital of France?")


    for question_text in questions:
        print(f"\n-----------------------------------------------------------")
        print(f"QUERYING RAG Chain with question: '{question_text}'")
        print(f"-----------------------------------------------------------")
        
        try:
            result = rag_chain.invoke({"question": question_text})

            print("\n>>> LLM Answer:")
            print(result.get("answer", "No answer found in result object."))
            
            print("\n>>> Retrieved Source Documents (for verification):")
            context_docs = result.get("context_docs", [])
            if context_docs:
                for i, doc in enumerate(context_docs):
                    print(f"\n  --- Document {i+1} ---")
                    print(f"  Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
                    print(f"  Content Snippet (first ~200 chars): {doc.page_content[:200]}...")
            else:
                print("  No context documents were retrieved or returned.")
        
        except Exception as e:
            print(f"An error occurred while invoking the RAG chain for question '{question_text}': {e}")
            import traceback
            traceback.print_exc()
        print(f"-----------------------------------------------------------\n")

if __name__ == "__main__":
    run_rag_test()