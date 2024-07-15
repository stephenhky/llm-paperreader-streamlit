
import os
import tempfile

import boto3
from botocore.config import Config
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_aws.llms.bedrock import BedrockLLM
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# load AWS tokens
load_dotenv()


# Streamlit page
st.set_page_config(page_title='LLM Paper Reader')

# set text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=20,
    length_function=len
)

llm_name = st.selectbox('Language Model', ['mistral.mixtral-8x7b-instruct-v0:1'])
max_tokens = st.number_input('max_tokens', min_value=0, value=4096)
temperature = st.number_input('temperature', min_value=0.0, value=0.7)
top_p = st.number_input('top_p', min_value=0.0, max_value=1.0, value=0.8)

# Get uploaded file
uploaded_pdffile = st.file_uploader('Upload a file (.pdf)')

# action
st.text('What to do?')
action = st.radio('Summarize', ['Summarize', 'Question & Answering'])

if (uploaded_pdffile is not None):
    pdfbytes = tempfile.NamedTemporaryFile()
    tempfilename = pdfbytes.name
    pdfbytes.write(uploaded_pdffile.read())

    st.text('Reading the file...')
    loader = PyPDFLoader(tempfilename)
    pages = loader.load_and_split(text_splitter=text_splitter)
    st.text('...done!')

    llm_config = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    aws_access_token = os.getenv('AWS_ACCESS_TOKEN')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=Config(read_timeout=1024),
        aws_access_key_id=aws_access_token,
        aws_secret_access_key=aws_secret_key
    )
    llm_model = BedrockLLM(model_id=llm_name, client=bedrock_runtime, config=llm_config)

    if action == 'Summarize':
        if st.button('Summarize'):
            chain = load_summarize_chain(llm=llm_model, chain_type='map_reduce')
            response = chain.run(pages)

            st.markdown(response)
    elif action == 'Question & Answering':
        st.text('Handling the file...')
        embeddings = GPT4AllEmbeddings(model_name='all-MiniLM-L6-v2.gguf2.f16.gguf')
        db = FAISS.from_documents(pages, embeddings)
        st.text('...done!')

        question = st.text_area('Ask a question:')
        to_ask = st.button('Ask')
        if to_ask:
            retriever = db.as_retriever()
            qa = RetrievalQA.from_chain_type(
                llm=llm_model,
                chain_type='stuff',
                retriever=retriever,
                return_source_documents=False
            )
            response_json = qa({'query': question})
            st.markdown(response_json['result'])
