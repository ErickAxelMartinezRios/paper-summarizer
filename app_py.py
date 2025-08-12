{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMpLHVp/zCNbicsZcd7xRfF",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ErickAxelMartinezRios/paper-summarizer/blob/main/app_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "pip install streamlit langchain huggingface_hub pymupdf langchain_community"
      ],
      "metadata": {
        "id": "2rCNRN_epAI1"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eQTxuEwId7pM",
        "outputId": "93a42a4a-f10b-4762-dcf7-beac5f5fb1d5"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-08-12 00:17:29.380 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.388 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.391 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.398 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.401 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.403 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.405 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.408 Session state does not function when running a script without `streamlit run`\n",
            "2025-08-12 00:17:29.410 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.412 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.416 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.417 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.419 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.422 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.433 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.434 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-12 00:17:29.437 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ],
      "source": [
        "import streamlit as st\n",
        "import fitz  # PyMuPDF\n",
        "from langchain import LLMChain, PromptTemplate\n",
        "from langchain_community.llms import HuggingFaceHub\n",
        "\n",
        "# ------------ Extract text from PDF -------------\n",
        "def extract_text_from_pdf(file):\n",
        "    doc = fitz.open(stream=file.read(), filetype=\"pdf\")\n",
        "    text = \"\"\n",
        "    for page in doc:\n",
        "        text += page.get_text()\n",
        "    return text\n",
        "\n",
        "# ------------ Summarize using HuggingFaceHub -------------\n",
        "def summarize_text(text, hf_token):\n",
        "    llm = HuggingFaceHub(\n",
        "        repo_id=\"facebook/bart-large-cnn\",\n",
        "        task=\"summarization\",\n",
        "        huggingfacehub_api_token=hf_token,\n",
        "        model_kwargs={\"max_length\": 256, \"temperature\": 0.0},\n",
        "    )\n",
        "    prompt = PromptTemplate(\n",
        "        input_variables=[\"content\"],\n",
        "        template=\"Summarize the following technical paper content clearly and concisely:\\n\\n{content}\"\n",
        "    )\n",
        "    chain = LLMChain(llm=llm, prompt=prompt)\n",
        "    limited_text = text[:1000]  # Avoid token limit errors\n",
        "    return chain.run(content=limited_text)\n",
        "\n",
        "# ------------- Streamlit UI -------------\n",
        "st.title(\"📄 Online Technical Paper Summarizer (Free)\")\n",
        "\n",
        "hf_token = st.text_input(\n",
        "    \"Enter your Hugging Face API token (get it from huggingface.co/settings/tokens):\",\n",
        "    type=\"password\"\n",
        ")\n",
        "\n",
        "uploaded_file = st.file_uploader(\"Upload a PDF file\", type=[\"pdf\"])\n",
        "\n",
        "if uploaded_file and hf_token:\n",
        "    st.info(\"Extracting text from PDF...\")\n",
        "    paper_text = extract_text_from_pdf(uploaded_file)\n",
        "\n",
        "    if st.button(\"Generate Summary\"):\n",
        "        st.info(\"Summarizing... please wait.\")\n",
        "        summary = summarize_text(paper_text, hf_token)\n",
        "        st.subheader(\"Summary\")\n",
        "        st.write(summary)\n",
        "elif uploaded_file and not hf_token:\n",
        "    st.warning(\"Please enter your Hugging Face API token to proceed.\")\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vYh6Rq1ghDXd",
        "outputId": "3776bfa6-172d-4928-e4b2-b9153958dfd6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-08-11 23:43:35.397 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-11 23:43:35.430 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2025-08-11 23:43:35.431 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-08-11 23:43:35.432 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    }
  ]
}