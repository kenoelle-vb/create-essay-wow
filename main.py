from langchain_community.document_loaders import TextLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader("dummy.txt")
data = loader.load()
st.write(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
chunks = text_splitter.split_documents(data)
st.write(chunks[0])
