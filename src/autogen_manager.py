# src/autogen_manager.py
import autogen
import json
# typing.List might not be needed if not used elsewhere now
# from typing import List

# --- REMOVE AutoGen Model Client Registration for LangchainLLM ---
# from autogen.oai.client import OpenAIWrapper 
# from autogen.oai.llm_utils import LangchainLLM 
# OpenAIWrapper.register_model_client(model_client_id="LangchainLLM", client_class=LangchainLLM)
# print("INFO: LangchainLLM registration lines removed.") 
# --- End REMOVE ---


# Config now directly uses OPENAI_API_KEY and the OpenAI model name
from .config import OPENAI_API_KEY, AUTOGEN_OPENAI_CONFIG_LIST 
from .autogen_tools import query_internal_knowledge_base, simple_calculator, perform_web_search

assistant_agent = None
user_proxy_agent = None

def get_autogen_agents():
    global assistant_agent, user_proxy_agent
    if assistant_agent is None or user_proxy_agent is None:
        print("Initializing AutoGen agents (Multi-Tool: RAG + Calc + WebSearch)...") # Updated
        
        llm_config_assistant = {
            "cache_seed": None, 
            "config_list": AUTOGEN_OPENAI_CONFIG_LIST, # Using OpenAI config
            "functions": [ # Define ALL available functions
                {
                    "name": "query_internal_knowledge_base",
                    # ... (description and parameters as before) ...
                    "description": "Queries internal company PDF documents for specific company information.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Question about company documents."}},
                        "required": ["query"]
                    }
                },
                {
                    "name": "simple_calculator",
                    # ... (description and parameters as before) ...
                    "description": "Evaluates basic arithmetic expressions.",
                     "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string", "description": "Arithmetic expression."}},
                        "required": ["expression"]
                    }
                },
                { # --- ADDED WEB SEARCH TOOL SCHEMA ---
                    "name": "perform_web_search",
                    "description": "Performs a web search to find up-to-date information, current events, or general knowledge not found in local documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for the web search."
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
        }
        
        assistant_agent = autogen.AssistantAgent(
            name="KnowledgeExplorerAssistant", # Updated name
            system_message=(
                "You are a helpful AI assistant with access to three tools: "
                "1. 'query_internal_knowledge_base': Use for questions about internal company documents (policies, product specs). "
                "2. 'simple_calculator': Use for mathematical calculations. "
                "3. 'perform_web_search': Use for current events, general knowledge, or if the user asks for up-to-date information not in local documents. "
                "Analyze the user's query carefully. "
                "If it's about company-specific information, call 'query_internal_knowledge_base'. "
                "If it's math, call 'simple_calculator'. "
                "If it requires current information or broader knowledge, call 'perform_web_search'. "
                "Do not make up answers for questions that should use a tool. "
                "After receiving a tool's result, present it clearly. "
                "For general conversation, answer directly. "
                "If unsure, you can state your available tools. "
                "After providing the complete answer, end your response with TERMINATE."
            ),
            llm_config=llm_config_assistant,
        )

        user_proxy_agent = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5, 
            is_termination_msg=lambda x: x.get("content", "") is not None and isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False, 
            function_map={ 
                "query_internal_knowledge_base": query_internal_knowledge_base,
                "simple_calculator": simple_calculator,
                "perform_web_search": perform_web_search # <-- ADDED MAPPING
            }
        )
        print("AutoGen agents initialized (RAG + Calc + WebSearch).") # Clarified print
    return assistant_agent, user_proxy_agent

# The if __name__ == "__main__": block remains the same.
if __name__ == "__main__":
    try:
        # We need OPENAI_API_KEY for this test scenario as Assistant uses OpenAI
        from src.config import OPENAI_API_KEY 
    except ImportError:
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.config import OPENAI_API_KEY


    print("Testing AutoGen Manager setup (Multi-Tool with OpenAI Assistant)...") # Clarified
    if not OPENAI_API_KEY: 
        print("OPENAI_API_KEY not set.")
    else:
        try:
            assistant, user_proxy = get_autogen_agents()
            if not assistant or not user_proxy:
                print("Failed to initialize AutoGen agents.")
                exit()
            
            print("AutoGen Assistant and UserProxy initialized successfully.")
            
            test_queries = [
                "What is the company policy on AI usage?", 
                "What is 125 divided by 5 plus 10?",
                "What's the latest news about SpaceX?",      
                "Hello there!",                           
                "Summarize the IT policy PDF and also tell me what 3*7 is." 
            ]

            for query_text in test_queries:
                print(f"\n\n--- Testing Query: '{query_text}' ---")
                user_proxy.reset() 

                user_proxy.initiate_chat(
                    assistant,
                    message=query_text,
                )
                last_message = user_proxy.last_message(assistant) # Get the last message this agent received from the assistant
                print(f"Assistant's final reply for '{query_text}':\n{last_message.get('content') if last_message else 'No final reply found.'}")
                print("--- End of Test Query ---")

        except Exception as e:
            print(f"ERROR during AutoGen Manager direct test: {e}")
            import traceback
            traceback.print_exc()