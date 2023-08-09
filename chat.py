# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
# from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

from langchain.chains import RetrievalQA
# from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

import os
# import nltk
import config
import logging
from typing import List
from langchain.llms import Replicate

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



def answer_llm_Faiss(prompt: str, documents: List[Document], persist_directory: str = config.PERSIST_DIR):

    from langchain.chains import RetrievalQA
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    
    from llm_wrapper.llm_wrapper import IdiomaLLM

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    # create the vectorestore to use as the index
    docsearch = FAISS.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":6})

    llm = IdiomaLLM()
    
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    #query = "How many AI publications in 2021?"
    result = qa({"query": prompt})

 #   prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])

    answer = result["result"]

    sources = result['source_documents']

    return answer, sources

def embed_document(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    # create the vectorestore to use as the index
    docsearch = FAISS.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":6})

    return retriever
    

def answer_replicate_Faiss(prompt: str, retriever, persist_directory: str = config.PERSIST_DIR):
        
    #from llm_wrapper.llm_wrapper import IdiomaLLM

    llm = Replicate(
        model="replicate/llama70b-v2-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48", #a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        input={"temperature": 0.75, "max_length": 500, "top_p": 1},
        )
    
   # from langchain import LLMChain
    # prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # {context}
    # Question: {question}
    # Answer in Italian:"""
    # PROMPT = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "question"]
    # )

    # prompt_template = """<s>[INST] <<SYS>>
    # You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    # If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    # <</SYS>>

    # [/INST]</s>
    # <s>[INST] 
    # {question}  
    # DOCUMENTS:
    # =========
    # {context}
    # ========= 
    #    [/INST]
    # """

    prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Context: {context}
        Question: {question}
        Only return the helpful answer below and nothing else.
        Helpful answer:
        """

    PROMPT_TEMPLATE = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # doc_chain = load_qa_chain(
    #     llm=llm,
    #     chain_type="stuff",
    #     prompt=PROMPT_TEMPLATE,
    # )

    # Log a message indicating the number of chunks to be considered when answering the user's query.
    LOGGER.info(f"The top {config.k} chunks are considered to answer the user's query.")

    qa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=retriever,
                                       chain_type_kwargs={'prompt': PROMPT_TEMPLATE})
            # return_source_documents=True,
    # Create a VectorDBQA object using a vector store, a QA chain, and a number of chunks to consider.
 #   qa = VectorDBQA(vectorstore=docsearch, combine_documents_chain=doc_chain, k=config.k)

    # Call the VectorDBQA object to generate an answer to the prompt.
    result1 = qa({"query": prompt})
    answer = result1["result"]


   # chain_type_kwargs = {"query": PROMPT_TEMPLATE}
    # chain_type_kwargs = {
    #     "llm_chain": llm_chain,
    #     "doc_retriever": retriever 
    # }

    # qa = RetrievalQA.from_chain_type(
    #     llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     chain_type_kwargs=chain_type_kwargs
    # )
  #  answer = qa.run(prompt)

    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    #query = "How many AI publications in 2021?"
    result = qa({"query": prompt})

 #   prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])

    #answer = result["result"]

    sources = result['source_documents']

    return answer, sources
