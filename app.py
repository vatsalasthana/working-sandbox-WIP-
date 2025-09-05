import streamlit as st
from frontend import run_frontend

if __name__ == "__main__":
    st.set_page_config(
        page_title="Brain-Heart AI System",
        page_icon="🧠❤️",
        layout="wide"
    )
    run_frontend()