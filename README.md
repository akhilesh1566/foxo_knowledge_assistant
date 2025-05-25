# Autogen Knowledge Assistant - Local Retrieval AI Agent

## Overview

The Autogen Knowledge Assistant is a AI agent designed to answer questions using internal company documents (PDFs, md and txt) and external tools. It leverages Retrieval-Augmented Generation (RAG) for document-based queries and can also perform calculations and web searches. The agent is built using Python, AutoGen for multi-agent orchestration, LangChain for RAG components, and Streamlit for a simple web interface.

## Setup Instructions

1. Prerequisites
*   Python 3.9 or higher (developed and tested with Python 3.11.8).
*   Git.

2. Clone the Repository
```bash
git https://github.com/akhilesh1566/foxo_knowledge_assistant.git
cd foxo_knowledge_assistant

3. Set Up a Virtual Environment
python -m venv venv
# Activate depends on your OS:
# Windows (Git Bash or PowerShell):
source venv/Scripts/activate
# macOS/Linux:
# source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

5. Configure Environment Variables (API Keys)
You will need API keys for OpenAI, Google (Gemini), and Tavily.

Create a .env file:
Copy the .env.example file to a new file named .env in the project root:
cp .env.example .env

Edit .env and add your API keys:
OPENAI_API_KEY="sk-your_openai_api_key_here"
GOOGLE_API_KEY="AIzaSyYour_google_api_key_here" 
TAVILY_API_KEY="tvly-your_tavily_api_key_here"

6. Prepare Documents for Ingestion
Place your PDF documents (at least 3 sample files are recommended for good testing) into the data/ directory in the project root.

7. Run the Application & Ingest Documents
Run the Streamlit application:

streamlit run app.py




Tools Used & Rationale:


AutoGen: Chosen for its powerful multi-agent capabilities, allowing for sophisticated tool orchestration and the creation of specialized agents. It handles the OpenAI function calling flow for tool selection effectively.
LangChain: Utilized for its robust RAG pipeline components:
Document loaders and text splitters for processing PDFs.
Integrations for Google Gemini embeddings (GoogleGenerativeAIEmbeddings).
ChromaDB vector store wrapper for easy interaction.
Prompt templates and LCEL (LangChain Expression Language) for constructing the RAG chain used by the internal knowledge base tool.
TavilySearchResults wrapper for quick web search integration.
OpenAI GPT Models: Used as the LLM for the primary AssistantAgent in AutoGen due to strong reasoning, instruction following, and reliable function calling capabilities.
Google Gemini Models:
models/embedding-001: For generating high-quality text embeddings for document chunks.
gemini-1.0-pro-latest (or similar): Used as the LLM within the RAG tool to synthesize answers specifically from the retrieved document context. This decouples the RAG answer synthesis from the main assistant's LLM.
ChromaDB: Selected as a simple, local, and open-source vector database, suitable for POCs and easy setup.
PyMuPDF: A fast and efficient library for PDF text and metadata extraction.
Streamlit: Chosen for its ability to quickly create interactive and user-friendly Python web applications with minimal front-end code.
Tavily API: Provides a simple and effective search API with a generous free tier, ideal for adding web search capabilities to a POC.

Agent Functioning:
Assistant decides to call a function, its message to UserProxy includes function_call.
UserProxy sees function_call, executes the function from function_map.
UserProxy sends a message with role function (or tool) containing the result back to Assistant.
Assistant uses this result to generate its final response.


Sample Questions & Responses:


1. Querying Internal Documents (RAG Tool):

KnowledgeExplorerAssistant: Hello! I am the Autogen Multi-Tool Assistant. I can search documents or perform calculations. How can I help?

User: when Third Party and Outsourcing Services Contract shall be awarded?

UserProxy: when Third Party and Outsourcing Services Contract shall be awarded?

assistant: 🛠️ Calling tool: query_internal_knowledge_base Arguments:

{
"query": "When shall Third Party and Outsourcing Services Contract be awarded?"
}

⚙️ Result from query_internal_knowledge_base:

Answer: Any Third Party and Outsourcing Services Contract shall be awarded after completion of the Vendor Evaluation and Selection process.
.

Cited Sources: [Source: ITPOLICYFINAL.pdf, Page: 56]; [Source: ITPOLICYFINAL.pdf, Page: 57]; [Source: ITPOLICYFINAL.pdf, Page: 58]

KnowledgeExplorerAssistant: The Third Party and Outsourcing Services Contract shall be awarded after completion of the Vendor Evaluation and Selection process.



2. Using the Calculator Tool:

User: What is (150 / 3) + 25 * 2?

UserProxy: What is (150 / 3) + 25 * 2?

assistant: 🛠️ Calling tool: simple_calculator Arguments:

{
"expression": "(150 / 3) + 25 * 2"
}

⚙️ Result from simple_calculator:

The result of '(150 / 3) + 25 * 2' is 100.0.



3. Using the Web Search Tool:

User: What is the latest news about renewable energy in Europe?

UserProxy: What is the latest news about renewable energy in Europe?

assistant: 🛠️ Calling tool: perform_web_search Arguments:

{
"query": "latest news about renewable energy in Europe"
}

⚙️ Result from perform_web_search:

Web Search Results:
1. [No Title](https://renewablesnow.com/regions/europe/)
   - Snippet: Comprehensive news coverage of renewable energy projects, investments, and policies across Europe. Track the latest in solar PV, onshore and offshore wind...
2. [No Title](https://www.euronews.com/tag/renewable-energies)
   - Snippet: Seven out of 10 countries set to exceed their wind power targets are in Europe, according to a new report. 12/08/2024. Subsidies caused solar panel...
3. [No Title](https://energy.ec.europa.eu/news/renewables-directive-sectoral-targets-reach-transposition-deadline-2025-05-21_en)
   - Snippet: Agreed by the co-legislators in 2023 and in force since 20 November 2023, the ultimate ambition of the directive is to achieve the EU's renewable target for 2030 of at least 42.5%, with the ambition to aim for 45%. This objective will require a conce...

KnowledgeExplorerAssistant: I found some recent news about renewable energy in Europe:

Renewablesnow: This website offers comprehensive news coverage of renewable energy projects, investments, and policies across Europe. You can track the latest developments in solar PV, onshore and offshore wind, and more.
Euronews: A report shows that seven out of 10 countries in Europe are set to exceed their wind power targets. Subsidies have influenced the growth of solar panel installations.
European Commission - Energy: The Renewables Directive includes sectoral targets to reach transposition by 2025. The EU aims to achieve renewable targets for 2030 with an ambition to reach at least 42.5% and potentially 45%.




Future Enhancements:

While getting data from pdf file we can extract images and tables for better understanding of context
Advanced External Tools:
Integrate safer math parsers or tools like Wolfram Alpha.
Use more specialized search APIs (e.g., news APIs, academic search).
Add tools for code execution (with appropriate sandboxing), interacting with calendars, or sending emails.
Sophisticated Agent Planning: Explore AutoGen's GroupChat or create a dedicated "Planner Agent" to decompose complex tasks and coordinate multiple specialized agents.
Improved Context Management: Implement more advanced strategies for managing conversation history and context windows, especially for long interactions.
User Feedback Mechanism: Allow users to rate answers or correct the agent's tool choices.




## Features

*   **Document Ingestion:** Processes local PDF files, chunks them, generates embeddings (using Google's Gemini embedding models), and stores them in a local ChromaDB vector database.
*   **Retrieval-Augmented Generation (RAG):** Retrieves relevant document snippets based on user queries and uses an LLM (Google Gemini model within the RAG tool) to synthesize answers from this context.
*   **Source Attribution:** Provides filename, page number, and context snippet for answers derived from internal documents.
*   **Multi-Agent System (AutoGen):**
    *   Employs an `AssistantAgent` (powered by OpenAI's GPT models for reasoning and tool selection) and a `UserProxyAgent`.
    *   Supports multi-step reasoning and intelligent tool routing.
*   **Multi-Tool Capabilities:**
    1.  **Internal Knowledge Base Query:** Accesses information from ingested PDFs.
    2.  **Simple Calculator:** Evaluates basic arithmetic expressions.
    3.  **Web Search:** Fetches up-to-date information from the internet using the Tavily Search API.
*   **User Interface:** A basic web UI built with Streamlit for interaction and document ingestion.

## Tech Stack

*   **Python:** 3.9+ (Developed with 3.11.8)
*   **Core AI/Agent Frameworks:**
    *   `pyautogen~=0.2.20`: For multi-agent orchestration and tool use.
    *   `langchain~=0.1.12`: For RAG pipeline components (document loading, splitting, embeddings, vector store integration, prompt templates).
*   **LLMs & Embeddings:**
    *   `openai~=1.14.3`: For the AutoGen Assistant Agent's reasoning and tool selection (e.g., GPT-3.5-Turbo, GPT-4-Turbo).
    *   `google-generativeai~=0.4.0` (via `langchain-google-genai~=0.0.9`): For generating text embeddings (e.g., `models/embedding-001`) and for the LLM used *within* the RAG tool to synthesize answers from document context (e.g., `gemini-1.0-pro-latest`).
*   **Vector Database:**
    *   `chromadb~=0.4.24`: For local storage and retrieval of document embeddings.
*   **Document Processing:**
    *   `PyMuPDF~=1.23.26`: For parsing PDF files.
*   **Web Search Integration:**
    *   `tavily-python`: For the web search tool.
*   **User Interface:**
    *   `streamlit~=1.31.0`: For building the interactive web application.
*   **Utilities:**
    *   `python-dotenv~=1.0.0`: For managing environment variables.
    *   `tiktoken~=0.5.2`: (Often a LangChain dependency for token counting).

## Project Structure

autogen_knowledge_assistant_gemini/
├── .env.example # Example environment variables file
├── .gitignore # Specifies intentionally untracked files that Git should ignore
├── app.py # Main Streamlit application file
├── ingest.py # Script for document ingestion pipeline
├── requirements.txt # Python package dependencies
├── README.md # This file
├── data/ # Folder to place local PDF documents for ingestion
│ └── (your_sample_pdfs.pdf)
├── src/ # Source code for the assistant's modules
│ ├── init.py
│ ├── autogen_manager.py # Configures and provides AutoGen agents
│ ├── autogen_tools.py # Defines tools callable by AutoGen agents (RAG, calc, web search)
│ ├── config.py # Loads API keys and central configuration
│ ├── document_processor.py # Handles PDF parsing and text chunking
│ ├── rag_chain_builder.py # Builds the LangChain RAG chain (used by the RAG tool)
│ └── vector_store_manager.py # Manages ChromaDB vector store and embeddings
└── vector_store/ # Directory where ChromaDB stores its data (created on first ingestion)

