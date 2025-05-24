# app.py
import streamlit as st
import os
from src.config import OPENAI_API_KEY # Using OPENAI_API_KEY for the assistant
from ingest import main as run_ingestion 
from src.autogen_manager import get_autogen_agents # This now sets up OpenAI assistant

# --- Page Configuration ---
st.set_page_config(
    page_title="FOXO Knowledge Assistant (AutoGen+OpenAI)",
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
if "display_chat_messages" not in st.session_state: # For displaying messages in Streamlit UI
    st.session_state.display_chat_messages = []


# --- Helper Functions ---
def initialize_autogen_agents_for_app():
    """Initializes AutoGen agents for the Streamlit app."""
    if st.session_state.user_proxy is None or st.session_state.assistant_agent is None:
        with st.spinner("Initializing Knowledge Assistant (AutoGen + OpenAI)... Please wait."):
            try:
                st.session_state.assistant_agent, st.session_state.user_proxy = get_autogen_agents()
                # Add initial greeting to display history after successful init
                if not st.session_state.display_chat_messages: # Only add if history is empty
                     st.session_state.display_chat_messages.append({
                        "role": "assistant", 
                        "name": st.session_state.assistant_agent.name if st.session_state.assistant_agent else "DocAssistant",
                        "content": "Hello! I am the FOXO Assistant. How can I help you with your documents today?"
                    })
                st.success("Knowledge Assistant initialized!")
            except Exception as e:
                st.error(f"Error initializing AutoGen agents: {e}")
                st.exception(e) # Show full traceback in Streamlit for debugging
                st.stop()
    return st.session_state.assistant_agent, st.session_state.user_proxy

def display_message_in_ui(msg_data):
    """
    Displays a message in the Streamlit chat UI.
    Parses AutoGen message structure for relevant content.
    """
    role_from_msg = msg_data.get("role", "system").lower()
    agent_name_from_msg = msg_data.get("name", role_from_msg)
    content_from_msg = msg_data.get("content", "")
    function_call_from_msg = msg_data.get("function_call", msg_data.get("tool_calls"))

    # Determine the role for st.chat_message icon/layout ("user" or "assistant")
    streamlit_display_role = "user" if role_from_msg == "user" else "assistant"

    with st.chat_message(streamlit_display_role): # This sets the icon and alignment
        # Now, construct the content to display
        display_parts = []
        
        # Add a header with the agent's actual name if it's not the generic 'user' or 'assistant'
        # or if you always want to show it.
        if agent_name_from_msg and agent_name_from_msg.lower() != streamlit_display_role:
            display_parts.append(f"**{agent_name_from_msg}:**")
        
        if function_call_from_msg:
            func_name = function_call_from_msg[0].get("name") if isinstance(function_call_from_msg, list) else function_call_from_msg.get("name")
            # func_args = function_call_from_msg[0].get("arguments") if isinstance(function_call_from_msg, list) else function_call_from_msg.get("arguments")
            display_parts.append(f"🛠️ *Using tool: `{func_name}`...*") # Args can be verbose for UI
            if content_from_msg and str(content_from_msg).strip() and str(content_from_msg).lower() != "none":
                display_parts.append(str(content_from_msg))
        elif role_from_msg == "function":
            display_parts.append(f"⚙️ **Result for `{agent_name_from_msg}`**:\n```text\n{str(content_from_msg)}\n```")
        elif content_from_msg and str(content_from_msg).strip():
            display_parts.append(str(content_from_msg))
        
        if display_parts:
            st.markdown("\n\n".join(display_parts))

# --- Main Application ---
st.title("🦊 FOXO Knowledge Assistant")
st.caption("Powered by AutoGen, OpenAI, and Google Gemini Embeddings")

if not OPENAI_API_KEY:
    st.error("🚨 OPENAI_API_KEY is not configured! This is needed for the AutoGen Assistant.")
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
    elif not vector_store_exists and not st.session_state.ingestion_done_first_time : # Only warn if not yet loaded
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
                    # Reset agents and chat so they pick up new knowledge or re-initialize states
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
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message to Streamlit display history and display it immediately
    st.session_state.display_chat_messages.append({"role": "user", "name": "User", "content": prompt})
    with st.chat_message("user"): # Display user's prompt
        st.markdown(prompt)

    # Perform AutoGen interaction
    # `assistant` and `user_proxy` are the initialized agents from initialize_autogen_agents_for_app()
    with st.spinner(f"{assistant.name if assistant else 'Assistant'} is thinking..."):
        try:
            if not assistant or not user_proxy: 
                st.error("Agents not initialized. Please refresh or check logs.")
                st.stop()

            user_proxy.reset() 
            # assistant.reset() # Optional

            user_proxy.initiate_chat(
                assistant, # <--- CHANGED assistant_agent to assistant
                message=prompt,
            )
            
            conversation_this_turn = user_proxy.chat_messages.get(assistant, []) # <--- CHANGED assistant_agent to assistant
            
            added_assistant_final_reply = False
            if conversation_this_turn:
                for msg_data_autogen in conversation_this_turn:
                    if msg_data_autogen.get("role") == "user" and msg_data_autogen.get("content") == prompt:
                        continue 

                    role_to_display = msg_data_autogen.get("role", "assistant")
                    if role_to_display == "model": role_to_display = "assistant"
                    
                    current_message_content = msg_data_autogen.get("content")
                    current_function_call = msg_data_autogen.get("tool_calls") or msg_data_autogen.get("function_call")

                    msg_for_streamlit_display = {
                        "role": role_to_display,
                        "name": msg_data_autogen.get("name", msg_data_autogen.get("role")),
                        "content": current_message_content,
                        "function_call": current_function_call
                    }

                    st.session_state.display_chat_messages.append(msg_for_streamlit_display)
                    
                    is_final_answer_candidate = (
                        role_to_display == "assistant" and 
                        current_message_content and 
                        isinstance(current_message_content, str) and 
                        current_message_content.strip() and
                        not current_function_call
                    )
                    
                    if is_final_answer_candidate:
                        added_assistant_final_reply = True
                        
            if not added_assistant_final_reply and conversation_this_turn:
                for msg_data_autogen in reversed(conversation_this_turn):
                    if msg_data_autogen.get("role") == "assistant" and \
                       msg_data_autogen.get("content") and \
                       isinstance(msg_data_autogen.get("content"), str) and \
                       not (msg_data_autogen.get("function_call") or msg_data_autogen.get("tool_calls")): # Ensure it's not a function call itself
                        added_assistant_final_reply = True
                        break
            
            if not added_assistant_final_reply: 
                 st.session_state.display_chat_messages.append({
                    "role": "assistant", 
                    "name": assistant.name if assistant else "DocAssistant", 
                    "content": "(Assistant processed the request but provided no further textual response for this turn.)"
                })

            st.rerun() 

        except Exception as e:
            error_msg = f"An error occurred during AutoGen interaction: {str(e)}"
            st.error(error_msg)
            st.exception(e)
            st.session_state.display_chat_messages.append({"role": "assistant", "name": "SystemError", "content": error_msg})
            st.rerun()