"""Browser-based Hania AI chat app."""

import streamlit as st

from app import GEMINI_BASE_URL, LocalBrain, OpenAIBrain, friendly_error, make_brain


st.set_page_config(page_title="Hania AI", page_icon="💬", layout="centered")

st.title("Hania AI")
st.caption("A friendly AI assistant")

with st.sidebar:
    st.subheader("About")
    st.write("Chat with Hania AI in your browser.")
    st.warning(
        "This app uses its owner's Gemini API quota. Anyone with the public link can send requests."
    )
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

try:
    api_key: str = str(st.secrets["GEMINI_API_KEY"]).strip()
except (KeyError, FileNotFoundError):
    st.error(
        "Gemini is not configured. Add GEMINI_API_KEY in the deployed app's private Secrets settings."
    )
    st.stop()

if not api_key:
    st.error("GEMINI_API_KEY is empty. Set it in the deployed app's private Secrets settings.")
    st.stop()

model: str = str(st.secrets.get("CUSTOM_MODEL_NAME", "gemini-3.8-flash")).strip()
base_url: str = str(st.secrets.get("CUSTOM_SERVER_URL", GEMINI_BASE_URL)).strip()

try:
    brain: LocalBrain | OpenAIBrain = make_brain("gemini", api_key, model, base_url)
except (ImportError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

if not isinstance(brain, OpenAIBrain):
    st.error("Gemini could not be initialized. Check the private app secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Message Hania AI"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response: str = st.write_stream(brain.stream(st.session_state.messages))
        except Exception as exc:
            st.error(friendly_error(exc))
        else:
            st.session_state.messages.append({"role": "assistant", "content": response})
