from langchain_community.document_loaders import TextLoader
import streamlit as st

loader = TextLoader("dummy.txt")
data = loader.load()
st.write(data)
