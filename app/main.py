import streamlit as st

from rag_pipeline import answer_question_with_rag
from pdf_processing import extract_text_from_pdf


st.title("AI PDF Chat App")

uploaded_file = st.file_uploader("Upload a sustainability report as PDF", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF uploaded successfully.")

    extracted_text = extract_text_from_pdf(uploaded_file)

    st.subheader("Extracted PDF Text Preview")

    if extracted_text:
        st.text_area("Preview", extracted_text[:3000], height=300)
    else:
        st.warning("No text could be extracted from this PDF.")

    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("Searching relevant PDF sections and generating answer with SAIA/Gemma..."):
            answer = answer_question_with_rag(question, extracted_text)

        st.subheader("Answer")
        st.write(answer)
else:
    st.info("Upload a PDF to start.")
