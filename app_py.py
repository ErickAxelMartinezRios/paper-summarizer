import streamlit as st
import fitz  # PyMuPDF
from langchain import LLMChain, PromptTemplate
from langchain_community.llms import HuggingFaceHub

# ------------ Extract text from PDF -------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ------------ Summarize using HuggingFaceHub -------------
def summarize_text(text, hf_token):
    llm = HuggingFaceHub(
        repo_id="facebook/bart-large-cnn",
        task="summarization",
        huggingfacehub_api_token=hf_token,
        model_kwargs={"max_length": 256, "temperature": 0.0},
    )
    prompt = PromptTemplate(
        input_variables=["content"],
        template="Summarize the following technical paper content clearly and concisely:\n\n{content}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    limited_text = text[:1000]  # Avoid token limit errors
    return chain.run(content=limited_text)

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
