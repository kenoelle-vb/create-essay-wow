from langchain_community.document_loaders import TextLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

loader = TextLoader("dummy.txt")
data = loader.load()
st.write(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
chunks = text_splitter.split_documents(data)
st.write(chunks[0])

llm = ChatGroq(groq_api_key="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", model_name="llama3-8b-8192")

#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

documents = chunks
embeddings = FastEmbedEmbeddings()
vector_store = Chroma.from_documents(documents, embeddings)
