from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import os
import nltk
import config
import logging
from typing import List

# Initialize logging with the specified configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_FILE),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

# Load documents from the specified directory using a DirectoryLoader object
#loader = DirectoryLoader(config.FILE_DIR, glob='*.pdf')
#documents = loader.load()
#documents = st.file_uploader("**Upload Your PDF File**", type=["pdf"])

# split the text to chuncks of of size 1000
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Split the documents into chunks of size 1000 using a CharacterTextSplitter object
#texts = text_splitter.split_documents(documents)

# Create a vector store from the chunks using an OpenAIEmbeddings object and a Chroma object
#embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
#docsearch = Chroma.from_documents(texts, embeddings)

# Define answer generation function
def answer(prompt: str, documents: List[Document], persist_directory: str = config.PERSIST_DIR):

    #documents = st.file_uploader("**Upload Your PDF File**", type=["pdf"])

    # split the text to chuncks of of size 1000
    text_splitter = CharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
    # Split the documents into chunks of size 1000 using a CharacterTextSplitter object
    texts = text_splitter.split_documents(documents)

    # Create a vector store from the chunks using an OpenAIEmbeddings object and a Chroma object
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    docsearch = Chroma.from_documents(texts, embeddings)

    # Log a message indicating that the function has started
    LOGGER.info(f"Start answering based on prompt: {prompt}.")

    # Create a prompt template using a template from the config module and input variables
    # representing the context and question.
    prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])

    # Load a QA chain using an OpenAI object, a chain type, and a prompt template.
    doc_chain = load_qa_chain(
        llm=OpenAI(
            openai_api_key = config.OPENAI_API_KEY,
            model_name="text-davinci-003",
            temperature=0,
            max_tokens=300,
            batch_size=5
        ),
        chain_type="map_reduce",
        prompt=prompt_template,
    )

    # Log a message indicating the number of chunks to be considered when answering the user's query.
    LOGGER.info(f"The top {config.k} chunks are considered to answer the user's query.")

    # Create a VectorDBQA object using a vector store, a QA chain, and a number of chunks to consider.
    qa = VectorDBQA(vectorstore=docsearch, combine_documents_chain=doc_chain, k=config.k)

    # Call the VectorDBQA object to generate an answer to the prompt.
    result = qa({"query": prompt})
    answer = result["result"]

    qa_sources = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key = config.OPENAI_API_KEY), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

    result_sources = qa_sources({"query": prompt})
    sources = result_sources['source_documents']

    # Log a message indicating the answer that was generated
    LOGGER.info(f"The returned answer is: {answer}")

    # Log a message indicating that the function has finished and return the answer.
    LOGGER.info(f"Answering module over.")
    return answer, sources



def answer_RetrievalQA(prompt: str, documents: List[Document], persist_directory: str = config.PERSIST_DIR):

    from langchain.chains import RetrievalQA
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    # create the vectorestore to use as the index
    docsearch = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":6})

    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(
            openai_api_key = config.OPENAI_API_KEY,
            model_name="text-davinci-003",
            temperature=0,
            max_tokens=300,
            batch_size=50,
        ), chain_type="map_reduce", retriever=retriever, return_source_documents=True)
    #query = "How many AI publications in 2021?"
    result = qa({"query": prompt})

 #   prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])

    answer = result["result"]

    sources = result['source_documents']

    return answer, sources


def answer_Faiss(prompt: str, documents: List[Document], persist_directory: str = config.PERSIST_DIR):

    from langchain.chains import RetrievalQA
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    # create the vectorestore to use as the index
    docsearch = FAISS.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":6})
    
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(
            openai_api_key = config.OPENAI_API_KEY,
            model_name="text-davinci-003",
            temperature=0,
            max_tokens=300,
            batch_size=50,
        ), chain_type="stuff", retriever=retriever, return_source_documents=True)
    #query = "How many AI publications in 2021?"
    result = qa({"query": prompt})

 #   prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])

    answer = result["result"]

    sources = result['source_documents']

    return answer, sources
