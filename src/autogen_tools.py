# src/autogen_tools.py
from .rag_chain_builder import get_rag_chain # RAG chain might still use Gemini internally
from typing import Annotated

_rag_chain_instance_for_tool = None

def get_rag_instance_for_tool():
    global _rag_chain_instance_for_tool
    if _rag_chain_instance_for_tool is None:
        print("Initializing RAG chain for AutoGen tool (internals could be Gemini)...")
        _rag_chain_instance_for_tool = get_rag_chain() # Uses GEMINI_CHAT_MODEL_FOR_RAG from config
    return _rag_chain_instance_for_tool

def query_internal_knowledge_base(
    query: Annotated[str, "The specific question to ask the internal knowledge base regarding company documents, policies, product details, etc."]
) -> Annotated[str, "The answer and source citations obtained from the internal knowledge base."]:
    """
    Queries the internal company knowledge base (PDF documents) to answer questions.
    Use this tool for questions about company policies, product specifications, internal reports, etc.
    """
    print(f"\n🤖 RAG TOOL CALLED (by OpenAI Assistant) with query: '{query}'")
    rag_chain = get_rag_instance_for_tool()
    if not rag_chain:
        return "Error: RAG chain for tool use is not available."
    try:
        result = rag_chain.invoke({"question": query})
        answer = result.get("answer", "No definitive answer found in the documents.")
        
        sources_summary = []
        if result.get("context_docs"):
            for doc in result.get("context_docs"):
                sources_summary.append(
                    f"[Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}]"
                )
        
        if sources_summary:
            return f"Answer: {answer}\nCited Sources: {'; '.join(sources_summary)}"
        else:
            return f"Answer: {answer} (No specific source documents were strongly matched by the RAG system for this query)."

    except Exception as e:
        print(f"Error in RAG tool execution: {e}")
        import traceback
        traceback.print_exc()
        return f"Sorry, an error occurred while querying the documents: {str(e)}"

if __name__ == '__main__':
    # Test the tool function directly
    print("Testing RAG tool function...")
    from src.config import GOOGLE_API_KEY
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("GOOGLE_API_KEY not set. Please configure .env file.")
    else:
        # Test with a sample query relevant to your ingested documents
        # sample_query = "What is the policy on acceptable use of IT resources?"
        sample_query = "How often should passwords be changed?"
        response = query_internal_knowledge_base(sample_query)
        print(f"\nResponse from RAG tool for query '{sample_query}':\n{response}")

        sample_query_no_answer = "What is the weather like today?"
        response_no_answer = query_internal_knowledge_base(sample_query_no_answer)
        print(f"\nResponse from RAG tool for query '{sample_query_no_answer}':\n{response_no_answer}")