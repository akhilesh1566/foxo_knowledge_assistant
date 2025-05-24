
from .rag_chain_builder import get_rag_chain 
from typing import Annotated
import traceback 
import os 

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    
    if os.getenv("TAVILY_API_KEY"):
        tavily_search_tool = TavilySearchResults(max_results=3) 
        print("INFO: Tavily Search tool initialized.")
    else:
        tavily_search_tool = None
        print("WARNING: Tavily Search tool not initialized due to missing TAVILY_API_KEY.")
except ImportError:
    print("WARNING: `langchain_community.tools.tavily_search` or `TavilySearchResults` not found. "
          "Ensure `tavily-python` and `langchain-community` are installed. Web search tool will be unavailable.")
    tavily_search_tool = None

_rag_chain_instance_for_tool = None

def get_rag_instance_for_tool():
    global _rag_chain_instance_for_tool
    if _rag_chain_instance_for_tool is None:
        print("INFO: Initializing RAG chain for AutoGen tool (internals could be Gemini)...")
        _rag_chain_instance_for_tool = get_rag_chain() 
    return _rag_chain_instance_for_tool

def query_internal_knowledge_base(
    query: Annotated[str, "The specific question to ask the internal knowledge base regarding company documents, policies, product details, etc."]
) -> Annotated[str, "The answer and source citations obtained from the internal knowledge base."]:
    """
    Queries the internal company knowledge base (PDF documents) to answer questions.
    Use this tool for questions about company policies, product specifications, internal reports, etc.
    """
    print(f"\n🤖 RAG TOOL CALLED (by Assistant) with query: '{query}'")
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
        traceback.print_exc()
        return f"Sorry, an error occurred while querying the documents: {str(e)}"

# --- NEW TOOL: Simple Calculator ---
def simple_calculator(
    expression: Annotated[str, "A basic arithmetic expression string to evaluate. For example, '2+2', '100 * 3.5 / 2', or '(5-3)*8'. Avoid complex functions or variables."]
) -> Annotated[str, "The result of the calculation or an error message."]:
    """
    A simple calculator that evaluates basic arithmetic expressions.
    Use this tool for mathematical calculations. Input must be a valid arithmetic expression string.
    It supports addition (+), subtraction (-), multiplication (*), and division (/).
    """
    print(f"\n🧮 CALCULATOR TOOL CALLED with expression: '{expression}'")
    try:
        # Basic safety: allow only numbers, operators, parentheses, and spaces.
        allowed_chars = "0123456789+-*/(). "
        if not all(char in allowed_chars for char in expression):
            return "Error: Expression contains invalid characters for simple_calculator."
        
        if any(keyword in expression.lower() for keyword in ['import', 'os', 'sys', 'eval', 'exec', 'lambda', 'def', '__']):
             return "Error: Expression contains disallowed keywords for safety."

        result = eval(expression)
        return f"The result of '{expression}' is {result}."
    except Exception as e:
        print(f"Error evaluating expression '{expression}' in simple_calculator: {e}")
        traceback.print_exc()
        return f"Error: Could not evaluate the expression '{expression}'. Ensure it's a valid simple arithmetic expression."

def perform_web_search(
    query: Annotated[str, "The search query for up-to-date information, current events, or general knowledge not found in local documents."]
) -> Annotated[str, "A concise summary of web search results or an error message if the search fails."]:
    """
    Performs a web search using Tavily to find up-to-date information or general knowledge.
    Use this tool when asked about current events, topics not covered by internal documents,
    or when explicitly asked to search the web.
    """
    print(f"\n🌐 WEB SEARCH TOOL CALLED with query: '{query}'")
    if not tavily_search_tool:
        return "Error: Tavily Search tool is not available or TAVILY_API_KEY is not set."
    
    try:
        
        results = tavily_search_tool.invoke(query) 

        if isinstance(results, list) and results: 
            summary = "Web Search Results:\n"
            for i, res_dict in enumerate(results):
                title = res_dict.get("title", "No Title")
                url = res_dict.get("url", "#")
                content_snippet = res_dict.get("content", "No snippet available.")
                summary += f"{i+1}. [{title}]({url})\n   - Snippet: {content_snippet[:250]}...\n" 
            return summary
        elif isinstance(results, str): 
            return f"Web Search Results:\n{results}"
        else:
            return "No relevant results found from web search or an unexpected result format was received."

    except Exception as e:
        print(f"Error during Tavily web search: {e}")
        traceback.print_exc()
        return f"Error performing web search: {str(e)}"

if __name__ == '__main__':
    print("Testing autogen_tools.py directly...")
    

    print("\nTesting Calculator tool...")
    print(f"Expression '2+2': {simple_calculator('2+2')}")
    print(f"Expression '100 / 4 * 2': {simple_calculator('100 / 4 * 2')}")
    print(f"Expression '(5-1) * 10': {simple_calculator('(5-1) * 10')}")
    print(f"Expression '3 / 0': {simple_calculator('3/0')}") 
    print(f"Expression 'import os': {simple_calculator('import os')}") 
    print(f"Expression 'a+b': {simple_calculator('a+b')}") 

    print("\nTesting Web Search tool...")
    if tavily_search_tool:
        print(f"Search for 'latest AI news': {perform_web_search('latest AI news')}")
        print(f"Search for 'capital of France': {perform_web_search('capital of France')}")
    else:
        print("Skipping Web Search tool test as it's not available.")
