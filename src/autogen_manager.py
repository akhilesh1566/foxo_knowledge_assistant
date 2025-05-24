# src/autogen_manager.py
import autogen
from typing import List # Keep if needed, but not directly for this simpler setup
import json # Keep for parsing any complex string outputs if necessary

# Remove previous registration attempts for LangchainLLM
# from autogen.oai.client import OpenAIWrapper
# from autogen.oai.llm_utils import LangchainLLM
# OpenAIWrapper.register_model_client(...) # REMOVE THIS
# print("INFO: LangchainLLM registration line removed/commented.")


from .config import OPENAI_API_KEY, AUTOGEN_OPENAI_CONFIG_LIST # Use OpenAI config
from .autogen_tools import query_internal_knowledge_base # Import the Python tool function

# --- Define AutoGen Agents ---
assistant_agent = None
user_proxy_agent = None

def get_autogen_agents():
    global assistant_agent, user_proxy_agent
    if assistant_agent is None or user_proxy_agent is None:
        print("Initializing AutoGen agents (using OpenAI for Assistant)...")
        
        # LLM Configuration for AssistantAgent (using OpenAI)
        llm_config_assistant = {
            "cache_seed": None, 
            "config_list": AUTOGEN_OPENAI_CONFIG_LIST, # Use the OpenAI config list
            "functions": [ # Define the available functions/tools for OpenAI function calling
                {
                    "name": "query_internal_knowledge_base",
                    "description": "Queries internal company PDF documents to answer questions about policies, products, reports etc. Returns the answer and sources.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's question to be answered from the documents."
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
        }
        
        assistant_agent = autogen.AssistantAgent(
            name="DocAssistantOpenAI",
            system_message=(
                "You are a helpful AI assistant. You have access to a function called 'query_internal_knowledge_base' "
                "to answer questions using internal company documents. "
                "If the user's question seems to require looking up company-specific information, "
                "you MUST call the 'query_internal_knowledge_base' function. "
                "Do not answer from your general knowledge if the question pertains to company documents. "
                "After receiving the result from the function, present it clearly to the user. "
                "If the function result indicates no answer or an error, relay that. "
                "For general conversation not related to documents, you can answer directly. "
                "After you have provided the complete answer to the user's current question, end your response with the exact word TERMINATE." # <-- ADDED THIS
            ),
            llm_config=llm_config_assistant,
        )

        user_proxy_agent = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            is_termination_msg=lambda x: isinstance(x.get("content"), str) and x.get("content", "").rstrip().endswith("TERMINATE"), # <--- MODIFIED HERE
            code_execution_config=False, 
            function_map={
                "query_internal_knowledge_base": query_internal_knowledge_base
            }
        )
        
        print("AutoGen agents initialized (using OpenAI for Assistant).")
    return assistant_agent, user_proxy_agent


if __name__ == "__main__":
    try:
        from src.config import OPENAI_API_KEY # For the check
    except ImportError:
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.config import OPENAI_API_KEY

    print("Testing AutoGen Manager setup (OpenAI Assistant)...")
    if not OPENAI_API_KEY: # Check for OpenAI key now
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
                "Hello there!", 
                "What is 2+2?" 
            ]

            for query_text in test_queries:
                print(f"\n\n--- Testing Query: '{query_text}' ---")
                user_proxy.reset() 

                # Initiate chat. The flow should be:
                # 1. UserProxy sends message to Assistant.
                # 2. Assistant decides if to call the function.
                #    - If yes, its message will contain 'function_call'.
                #    - UserProxy's function_map executes it.
                #    - UserProxy sends result back to Assistant.
                # 3. Assistant formulates final reply to UserProxy.
                user_proxy.initiate_chat(
                    assistant,
                    message=query_text,
                )
                # The conversation (including function calls and results) will be printed to console by AutoGen.
                # We are interested in the final reply from the assistant.
                last_message = user_proxy.last_message(assistant) # Get the last message an agent sent
                print(f"Assistant's final reply for '{query_text}':\n{last_message.get('content') if last_message else 'No final reply found.'}")
                print("--- End of Test Query ---")

        except Exception as e:
            print(f"ERROR during AutoGen Manager direct test: {e}")
            import traceback
            traceback.print_exc()