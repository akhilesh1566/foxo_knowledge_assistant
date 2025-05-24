
import streamlit as st
import os
import json
from src.config import OPENAI_API_KEY 
from ingest import main as run_ingestion 
from src.autogen_manager import get_autogen_agents 

# --- Page Configuration ---
st.set_page_config(
    page_title="Autogen Knowledge Assistant (Multi-Tool)", # Updated title
    page_icon="🦊",
    layout="wide"
)

# --- Global State (Session State) ---
if "user_proxy" not in st.session_state:
    st.session_state.user_proxy = None
if "assistant_agent" not in st.session_state:
    st.session_state.assistant_agent = None
if "ingestion_done_first_time" not in st.session_state:
    st.session_state.ingestion_done_first_time = False
if "display_chat_messages" not in st.session_state: 
    st.session_state.display_chat_messages = []


# --- Helper Functions ---
def initialize_autogen_agents_for_app():
    if st.session_state.user_proxy is None or st.session_state.assistant_agent is None:
        with st.spinner("Initializing Knowledge Assistant... Please wait."):
            try:
                st.session_state.assistant_agent, st.session_state.user_proxy = get_autogen_agents()
                if not st.session_state.display_chat_messages:
                     st.session_state.display_chat_messages.append({
                        "role": "assistant", 
                        "name": st.session_state.assistant_agent.name if st.session_state.assistant_agent else "Assistant",
                        "content": "Hello! I am the Autogen Multi-Tool Assistant. I can search documents or perform calculations. How can I help?"
                    })
                st.success("Knowledge Assistant initialized!")
            except Exception as e:
                st.error(f"Error initializing AutoGen agents: {e}")
                st.exception(e) 
                st.stop()
    return st.session_state.assistant_agent, st.session_state.user_proxy

def display_message_in_ui(msg_data):
    role_from_msg = msg_data.get("role", "system").lower()
    agent_name_from_msg = msg_data.get("name", role_from_msg)
    content_from_msg = msg_data.get("content", "")
    
    function_call_details = msg_data.get("function_call") 
    is_function_result = role_from_msg == "function" 

    streamlit_display_role = "user" if role_from_msg == "user" else "assistant"

    with st.chat_message(streamlit_display_role): 
        display_header = f"**{agent_name_from_msg}:**"
        
        if function_call_details:
            func_name = function_call_details.get("name")
            try:
                # Arguments are a JSON string, parse for prettier display
                func_args_str = function_call_details.get("arguments", "{}")
                func_args_dict = json.loads(func_args_str)
                func_args_pretty = json.dumps(func_args_dict, indent=2)
                tool_info = f"🛠️ Calling tool: `{func_name}`\n   Arguments:\n   ```json\n{func_args_pretty}\n   ```"
            except json.JSONDecodeError:
                tool_info = f"🛠️ Calling tool: `{func_name}` with arguments: `{function_call_details.get('arguments')}`"
            
            if content_from_msg and str(content_from_msg).strip().lower() != "none":
                st.markdown(f"{display_header}\n{str(content_from_msg)}\n\n{tool_info}")
            else:
                st.markdown(f"{display_header}\n{tool_info}")

        elif is_function_result:
            st.markdown(f"⚙️ **Result from `{agent_name_from_msg}`**:\n```text\n{str(content_from_msg)}\n```")
        
        elif content_from_msg and str(content_from_msg).strip():
            processed_content = str(content_from_msg)
            if processed_content.rstrip().endswith("TERMINATE"):
                processed_content = processed_content.rstrip()[:-9].rstrip()
            
            if processed_content.strip(): 
                 st.markdown(f"{display_header}\n{processed_content}")

# --- Main Application ---
st.title("🦊 Autogen Knowledge Assistant")
st.caption("Multi-Tool: Document Q&A & Calculator") # Updated caption

if not OPENAI_API_KEY:
    st.error("🚨 OPENAI_API_KEY is not configured!")
    st.markdown("Please ensure your `.env` file in the project root is correctly set up.")
    st.markdown("After setting the key, you might need to **restart this Streamlit app**.")
    st.stop()

# --- Sidebar for Ingestion ---
with st.sidebar:

    st.header("📄 Document Management")
    vector_store_exists = os.path.exists(os.path.join("vector_store", "chroma.sqlite3"))
    
    if vector_store_exists and not st.session_state.ingestion_done_first_time:
        st.success("Knowledge base (vector store) seems to be loaded.")
        st.session_state.ingestion_done_first_time = True
    elif not vector_store_exists and not st.session_state.ingestion_done_first_time : 
        st.warning("Knowledge base not found. Please ingest documents if needed.")

    if st.button("🔄 Ingest/Re-Ingest Documents", help="Clears existing knowledge base and re-processes PDFs from the 'data' folder."):
        data_folder = "data"
        if not os.path.exists(data_folder) or not any(f.lower().endswith(".pdf") for f in os.listdir(data_folder)):
            st.error(f"No PDF files found in the '{data_folder}' directory. Please add some PDF documents there.")
        else:
            with st.spinner("Processing documents... This may take a few minutes."):
                try:
                    run_ingestion() 
                    st.success("✅ Documents ingested successfully!")
                    st.session_state.user_proxy = None 
                    st.session_state.assistant_agent = None
                    st.session_state.display_chat_messages = [] 
                    st.rerun() 
                except Exception as e:
                    st.error(f"An error occurred during ingestion: {e}")
                    st.exception(e)
    st.markdown("---")
    st.caption("Place PDF files in the 'data' folder.")


# --- Initialize AutoGen Agents ---
assistant, user_proxy = initialize_autogen_agents_for_app()
if not assistant or not user_proxy:
    st.warning("AutoGen agents could not be initialized. Please check errors above.")
    st.stop()

# --- Display Chat History ---
for msg in st.session_state.display_chat_messages:
    display_message_in_ui(msg)

# --- User Input and Chat Logic ---
if prompt := st.chat_input("Ask about documents or calculate something..."): # Updated placeholder
    st.session_state.display_chat_messages.append({"role": "user", "name": "User", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner(f"{assistant.name} is working..."):
        try:
            user_proxy.reset() 

            user_proxy.initiate_chat(
                assistant,
                message=prompt,
            )
            
            conversation_this_turn = user_proxy.chat_messages.get(assistant, [])
            
            if conversation_this_turn:
                new_messages_to_add_to_display = []
                for autogen_msg_data in conversation_this_turn:
                    # Avoid re-displaying the user's initial prompt if AutoGen includes it as the first message
                    if autogen_msg_data.get("role") == "user" and autogen_msg_data.get("content") == prompt:
                        if not new_messages_to_add_to_display: # Only skip if it's the very first message
                            continue
                    
                    # Prepare message for Streamlit display
                    msg_for_streamlit = {
                        "role": autogen_msg_data.get("role", "assistant"), # Default to assistant
                        "name": autogen_msg_data.get("name", autogen_msg_data.get("role")),
                        "content": autogen_msg_data.get("content"),
                        "function_call": autogen_msg_data.get("tool_calls") or autogen_msg_data.get("function_call")
                    }
                    new_messages_to_add_to_display.append(msg_for_streamlit)
                
                # Extend the display_chat_messages only with new messages from this turn
                st.session_state.display_chat_messages.extend(new_messages_to_add_to_display)
            else: # Should not happen if initiate_chat ran, but as a fallback
                 st.session_state.display_chat_messages.append({
                    "role": "assistant", 
                    "name": assistant.name, 
                    "content": "(No response from assistant for this turn.)"
                })
            
            st.rerun()

        except Exception as e:
            error_msg = f"An error occurred during AutoGen interaction: {str(e)}"
            st.error(error_msg)
            st.exception(e)
            st.session_state.display_chat_messages.append({"role": "assistant", "name": "SystemError", "content": error_msg})
            st.rerun()