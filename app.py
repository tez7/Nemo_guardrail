"""
Simple Streamlit chatbot powered by:
- Groq API (OpenAI-compatible)
- NVIDIA NeMo Guardrails
- Pydantic for validation & settings
"""

import os
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

from src.nemo.model import AppSettings, ChatHistory, ChatMessage, Role
from src.nemo.guardrail import GuardrailsChatbot

st.set_page_config(
    page_title="Groq + NeMo Guardrails Chatbot",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stChatMessage { padding: 0.75rem 1rem; }
    .main-header { text-align: center; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_guardrails(config_path: str = "config") -> GuardrailsChatbot:
    return GuardrailsChatbot(config_path=config_path)


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = ChatHistory()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    st.markdown(
        '<div class="main-header"><h1>🛡️ Groq + NeMo Guardrails</h1>'
        "<p>A simple, safety-guarded chatbot powered by Groq</p></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        st.markdown(
            """
            **Stack**
            - LLM: Groq (OpenAI-compatible)
            - Guardrails: NVIDIA NeMo Guardrails
            - Validation: Pydantic
            - UI: Streamlit
            """
        )
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.history.clear()
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption(
            "Guardrails active:\n"
            "- Input self-check\n"
            "- Output self-check\n"
            "- Basic dialog flows"
        )
        st.markdown("---")
        st.caption("Get your API key at [console.groq.com](https://console.groq.com)")

    try:
        settings = AppSettings()
        os.environ["OPENAI_API_KEY"] = os.environ["GROQ_API_KEY"]
        
    except Exception as e:
        st.error(f"⚠️ Configuration error:\n\n{e}")
        st.info(
            "1. Copy `.env.example` → `.env`\n"
            "2. Add your GROQ_API_KEY from https://console.groq.com\n"
            "3. Restart the app"
        )
        st.stop()

    init_session_state()

    try:
        chatbot = load_guardrails(settings.config_path)
    except Exception as e:
        st.error(f"Failed to load NeMo Guardrails:\n\n{e}")
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your message..."):
        try:
            user_msg = ChatMessage(role=Role.USER, content=prompt)
        except Exception as e:
            st.error(f"Invalid message: {e}")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_msg.content})
        with st.chat_message("user"):
            st.markdown(user_msg.content)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with guardrails..."):
                try:
                    response = chatbot.generate(
                        user_message=user_msg.content,
                        history=st.session_state.history,
                    )
                except Exception as e:
                    response = f"Sorry, something went wrong: {e}"
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.history.add_user_message(user_msg.content)
        st.session_state.history.add_assistant_message(response)


if __name__ == "__main__":
    main()