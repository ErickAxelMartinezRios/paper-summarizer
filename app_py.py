import streamlit as st
import fitz  # PyMuPDF
from langchain import LLMChain, PromptTemplate
from langchain_community.llms import HuggingFaceHub
from huggingface_hub import InferenceClient
# ------------ Extract text from PDF -------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ------------ Summarize using HuggingFaceHub -------------
def summarize_text(text, hf_token):
    client = InferenceClient(token=hf_token)
    # Use the bart-large-cnn model for summarization
    response = client.text_summarization(
        model="facebook/bart-large-cnn",
        inputs=text[:1000]  # limit length
    )
    return response[0]['summary_text']

# ------------- Streamlit UI -------------
st.title("📄 Online Technical Paper Summarizer (Free)")

hf_token = st.text_input(
    "Enter your Hugging Face API token (get it from huggingface.co/settings/tokens):",
    type="password"
)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file and hf_token:
    st.info("Extracting text from PDF...")
    paper_text = extract_text_from_pdf(uploaded_file)

    if st.button("Generate Summary"):
        st.info("Summarizing... please wait.")
        summary = summarize_text(paper_text, hf_token)
        st.subheader("Summary")
        st.write(summary)
elif uploaded_file and not hf_token:
    st.warning("Please enter your Hugging Face API token to proceed.")
