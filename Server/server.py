import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader

app = Flask(__name__)

global_url = ''
vectordb = None  # Initialize vectordb as None


def createDoc(url) :
    global global_url, vectordb
    loader = WebBaseLoader(url)
    data = loader.load_and_split()
    return data




def getResponse(url, prompt):
    global global_url, vectordb

    # Loading environment variables from .env file
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    system_template = """Use the following pieces of context to answer the users question or summarize the pieces of context.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    chain_type_kwargs = {"prompt": prompt}

    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "dbChroma")

    # Load data from the specified URL
    loader = WebBaseLoader(url)
    data = loader.load()

    # Split the loaded data
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size=500,
                                          chunk_overlap=40)
    docs = text_splitter.split_documents(data)

    if url != global_url:
        global_url = url

    response = generateResponse(docs, prompt, DB_DIR)

    return response


def generateResponse(docs, prompt, DB_DIR):
    global vectordb
    openai_embeddings = OpenAIEmbeddings()

    if vectordb is None:
        vectordb = Chroma(persist_directory=DB_DIR)

    for collection in vectordb._client.list_collections():
        ids = collection.get()['ids']
        if len(ids):
            collection.delete(ids)
        vectordb.persist()

    # Create a Chroma vector database from the documents
    print(docs)
    vectordb = Chroma.from_documents(documents=docs,
                                     embedding=openai_embeddings,
                                     persist_directory=DB_DIR)
    vectordb.persist()

    # Create a retriever from the Chroma vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Use a ChatOpenAI model
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')

    # Create a RetrievalQA from the model and retriever
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Run the prompt and return the response
    response = qa(prompt)

    return response


def getResponsePdf(prompt, text):
    global global_url, vectordb

    # Loading environment variables from .env file
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    system_template = """Use the following pieces of context to answer the users question or summarize the pieces of context.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    chain_type_kwargs = {"prompt": prompt}

    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "dbChroma")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_text(text=text)
    chunks = text_splitter.create_documents(splits)
    response = generateResponse(chunks, prompt, DB_DIR)

    return response


@app.route('/', methods=['POST'])
def handle_upload():
    # Read the incoming POST data
    text_data = request.form['prompt']

    # Access file data
    pdf_file = request.files.get('pdf')
    # Additional data (if any)
    url = request.form['url']
    print("Text data:", text_data)
    print("URL:", url)

    if pdf_file is None:
        response_data = getResponse(url, text_data)
        # Send the response as JSON
        print(response_data)
        return jsonify(response_data)

    if pdf_file.filename:
        pdf_reader = PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        print("Text extracted from PDF:", text)
        response_data = getResponsePdf(text_data, text)
        return jsonify(response_data)

    return 'Data received successfully'


if __name__ == '__main__':
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.join(ABS_PATH, "dbChroma")
    vectordb = Chroma(persist_directory=DB_DIR)
    app.run(debug=True, port=8000, host="localhost")
