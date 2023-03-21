import logging
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from langchain import OpenAI,VectorDBQA
import config
import fitz
import openai
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_FILE),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

file = config.FILE
# Load PDF document and split into chunks
doc = fitz.open(file)
text = ""
for page in doc:
    text += page.get_text()
#print(text)
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1300, chunk_overlap=0)
chunks = text_splitter.split_text(text)

#metadatas = [{"document_id": f"doc{i}", "author": "Guide Allianz"} for i in range(1, 96)]
num_docs = 95
metadata = []
for i in range(num_docs):
    doc_metadata = {"document_id": f"doc{i+1}", "Info": "CV - Abonia"}
    metadata.append(doc_metadata)
# Embed chunks and store in Chroma vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)

if  not os.path.exists(config.PERSIST_DIR):
    vectorstores = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=config.PERSIST_DIR, metadatas=None)
    LOGGER.info(f"All documents were processed and saved in {config.PERSIST_DIR}.")
else:
    vectorstore = Chroma(persist_directory=config.PERSIST_DIR, embedding_function=embeddings)

# Define answer generation function
def answer(prompt: str, persist_directory: str = config.PERSIST_DIR) -> str:
    LOGGER.info(f"Start answering based on prompt: {prompt}.")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])
    doc_chain = load_qa_chain(
        llm=OpenAI(
            openai_api_key = config.OPENAI_API_KEY,
            model_name="text-davinci-003",
            temperature=0,
            max_tokens=300,
        ),
        chain_type="stuff",
        prompt=prompt_template,
    )
    LOGGER.info(f"The top {config.k} chunks are considered to answer the user's query.")
    qa = VectorDBQA(vectorstore=vectorstore, combine_documents_chain=doc_chain, k=config.k)
    result = qa({"query": prompt})
    answer = result["result"]
    LOGGER.info(f"The returned answer is: {answer}")
    LOGGER.info(f"Answering module over.")
    return answer

"""import streamlit as st

# Create the Streamlit app
def main():
    # Add a title to the app
    st.title("Document-Based Chatbot")

    # Add an input box for the user to type their question
    user_input = st.text_input("Please enter your question:")

    # Add a button to submit the question
    if st.button("Submit"):
        # Call the answer function with the user's question as input
        response = answer(user_input)

        # Display the answer to the user
        st.write(response)

# Run the app
if __name__ == "__main__":
    main()"""

