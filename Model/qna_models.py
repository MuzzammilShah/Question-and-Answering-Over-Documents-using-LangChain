import streamlit as st
from transformers import AutoModelForSequenceClassification

import os
os.environ["OPENAI_API_KEY"] = "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"
openai_api_key='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'

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

# The vectorstore we'll be using
from langchain.vectorstores import FAISS

# The LangChain component we'll use to get the documents
from langchain.chains import RetrievalQA

# The embedding engine that will convert our text to vectors
from langchain.embeddings.openai import OpenAIEmbeddings

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Title
st.title('Welcome to MLQA!')
st.text("Muhammeds LLM Q&A App")

# Text input
model_input = st.text_area("Enter your input text:")

# Model selection
model_selection = st.selectbox("Select the model:", ["OpenAI Model", "Google LLM"])

from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('../', glob="**/*.txt")
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)

# Getting the embeddings engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embedding the documents and combining them with the raw text in a pseudo db. Note: This will make an API call to OpenAI
docsearch = FAISS.from_documents(docs, embeddings)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

if model_selection == "OpenAI Model":
    model_name = "distilbert-base-cased-distilled-squad"
else:
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Submit button
if st.button("Submit"):
    if model_selection == "OpenAI Model":
        st.text(qa.run(model_input))

    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        answer = model.predict(model_input)
        st.success(answer)