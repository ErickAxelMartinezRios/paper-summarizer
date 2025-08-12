import streamlit as st
import fitz
from huggingface_hub import InferenceClient

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def summarize_text(text, hf_token):
    client = InferenceClient(model="facebook/bart-large-cnn", token=hf_token)
    response = client.summarization(text[:1000])
    if isinstance(response, list) and len(response) > 0:
        return response[0].get("summary_text", "No summary returned.")
    return "No summary returned."

st.title("📄 Technical Paper Summarizer")

hf_token = st.text_input("Enter your Hugging Face API token:", type="password")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file and hf_token:
    st.info("Extracting text from PDF...")
    paper_text = extract_text_from_pdf(uploaded_file)
    st.write("Extracted text snippet:", paper_text[:500])
    if st.button("Generate Summary"):
        st.info("Summarizing...")
        summary = summarize_text(paper_text, hf_token)
        st.subheader("Summary")
        st.write(summary)
elif uploaded_file and not hf_token:
    st.warning("Please enter your Hugging Face API token to continue.")
