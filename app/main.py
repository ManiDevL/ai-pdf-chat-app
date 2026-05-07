import streamlit as st

st.title("AI PDF Chat App")

uploaded_file = st.file_uploader("Upload a sustainability report as PDF", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF uploaded successfully.")

question = st.text_input("Ask a question about the PDF")

if question:
    st.write("Answer will be generated here later.")
