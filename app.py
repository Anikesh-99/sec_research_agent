"""
Streamlit front end. Ask a question, get a cited answer.

Run:
  streamlit run app.py

Assumes you've already indexed at least one ticker:
  python -m index.build_index NVDA
"""

import streamlit as st

from agent.research_agent import ask

st.set_page_config(page_title="SEC Filing Research Agent", layout="wide")
st.title("SEC Filing Research Agent")
st.caption("Ask about a company's 10-K / 10-Q. Answers cite the filing and section.")

with st.sidebar:
    st.markdown("**Indexed filings live in `data/chroma/`.**")
    st.markdown("Add a company first:")
    st.code("python -m index.build_index NVDA", language="bash")

question = st.text_input(
    "Question",
    placeholder="What are the main risk factors in NVDA's latest 10-K?",
)

if st.button("Ask") and question:
    with st.spinner("Researching filings..."):
        answer = ask(question)
    st.markdown(answer)
