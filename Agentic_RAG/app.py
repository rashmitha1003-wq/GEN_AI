import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from main import build_agent, MEDICAL_PDF_PATH, EXERCISE_PDF_PATH

# Load environment
load_dotenv()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Health Assistant", layout="wide")
st.title("⚕️ Health Assistant")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    st.info("API keys are loaded from your .env file. Using local PDFs: `medical.pdf` and `physical activity.pdf`.")
    process_button = st.button("Initialize Agent")

# --- Session State ---
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "app" not in st.session_state:
    st.session_state.app = None

# --- Initialize Agent ---
if process_button:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API Key not found. Please set it in your .env file.")
    elif not os.path.exists(MEDICAL_PDF_PATH) or not os.path.exists(EXERCISE_PDF_PATH):
        st.error(f"PDFs not found. Ensure '{MEDICAL_PDF_PATH}' and '{EXERCISE_PDF_PATH}' exist.")
    else:
        with st.spinner("Processing documents and setting up the agent..."):
            st.session_state.app = build_agent(logger=st.write)
            if st.session_state.app:
                st.session_state.agent_ready = True
                st.success("✅ Agent is ready! Ask your health-related questions below.")
            else:
                st.error("❌ Failed to initialize the agent.")

# --- Chat Interface ---
if st.session_state.agent_ready:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your health question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                state = {"messages": [HumanMessage(content=prompt)]}
                result = st.session_state.app.invoke(state)
                response = result['messages'][-1].content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please click **Initialize Agent** in the sidebar to start.")