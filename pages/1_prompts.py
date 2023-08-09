import streamlit as st
from db import retrieve_logs
import sqlite3
import json

st.set_page_config(layout="wide", page_title="Data Analytics: Prompt DB", page_icon="📝")

st.markdown("""
<h1 style='text-align: center;'>Prompt DB</h1>
<style>
    .katex .base {
        width: 100%;
        display: flex;
        flex-wrap: wrap;
    }
    .stCodeBlock code {
        white-space: break-spaces !important;
        }
</style>
""", unsafe_allow_html=True)

st.markdown("---")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        _, col, _ = st.columns(3)
        with col:
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        _, col, _ = st.columns(3)
        with col:
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    col1, col2, col3 = st.columns((4, 48, 48))

    with col1:
        st.markdown("**ID**")
    with col2:
        st.markdown("**Prompt**")
    with col3:
        st.markdown("**Answer**")

    for log in retrieve_logs():
        id, prompt, answer, citations = log

        citations = json.loads(citations)

        with col1:
            st.markdown(f"### {id}")
        with col2: 
            st.info(prompt)
        with col3:
            st.success(answer)

        c1, c2 = st.columns((4, 96))
        with c2:
            with st.expander("Citations"):
                for citation in citations:
                    st.warning(citation)

        st.markdown("---")

    
