import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

st.title("Article Research Tool")

st.sidebar.title("Upload URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Create Knowledge Base")

# Create LLM
llm = OpenAI(temperature = 0.9, max_tokens = 500)

main_placeholder = st.empty()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls = urls)
    
    main_placeholder.text("Loading data...")

    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )

    docs = text_splitter.split_documents(data)

    # create embeddings and save to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building Embedding Vector Database...")
    vectorindex_openai.save_local("faiss_store")

query = main_placeholder.text_input("Question: ")
if query:
    vector_store = FAISS.load_local("faiss_store", OpenAIEmbeddings())
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)
    st.header("Answer")
    st.subheader(result["answer"])

    # Display sources
    sources = result.get("sources", "")
    if sources:
        st.write("Sources:")
        sources_list = sources.split('\n')
        for source in sources_list:
            st.write(source)