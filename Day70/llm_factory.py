
import os

from config import IS_CLOUD, LLM_MODEL


def get_chat_model():

    if IS_CLOUD:
        from langchain_groq import ChatGroq

        # Read the key from Streamlit Secrets or environment.
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                api_key = None

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
            )

        return ChatGroq(
            model=LLM_MODEL,
            temperature=0,
            api_key=api_key
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
        num_predict=512
    )
