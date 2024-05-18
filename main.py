import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("ResearchBOT: Article Research Tool")
st.sidebar.title("Articles URL")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.7, max_tokens=500)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading.... Started...")
    data = loader.load()
    if not data:
        st.error("Failed to load data from URLs. Please check the URLs and try again.")
    else:
        main_placeholder.text("Data Loaded Successfully.")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter .... Started ...")
        docs = text_splitter.split_documents(data)
        if not docs:
            st.error("Failed to split documents. Please check the data and try again.")
        else:
            main_placeholder.text(f"Documents split into {len(docs)} chunks.")

            embeddings = OpenAIEmbeddings()
            doc_embeddings = embeddings.embed_documents(docs)
            if not doc_embeddings:
                st.error("Failed to create embeddings. Please check your OpenAI API key and try again.")
            else:
                main_placeholder.text("Embedding Vector Started Building")
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                time.sleep(2)

                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_openai, f)
                main_placeholder.text("Vector store created and saved successfully.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                st.write(sources_list)
