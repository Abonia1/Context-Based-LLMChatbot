import chat
import streamlit as st
from streamlit_chat import message
import re
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Iterator
from langchain.schema import Document
from pypdf import PdfReader
import textwrap
import os
#import replicate
import config
from db import *
from st_pages import hide_pages
import os

st.set_page_config(page_title="DOCCHAT | WITHMOBIUS", 
                #    initial_sidebar_state="collapsed",
                   page_icon="data:image/png;base64,/9j/4AAQSkZJRgABAQIAHAAcAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAAgACADAREAAhEBAxEB/8QAGQAAAgMBAAAAAAAAAAAAAAAABAcFBggJ/8QAMBAAAgEDAwEECAcAAAAAAAAAAQIDBAURAAYSIQcIIjETFBUyQVFhcSNDYoGRkqH/xAAZAQACAwEAAAAAAAAAAAAAAAACBAEFBgD/xAAlEQACAgECBgMBAQAAAAAAAAABAgADBBEhBRITIlGBMUGRMtH/2gAMAwEAAhEDEQA/AOVWunSfoOz7fd0jEtu2be6mMjkGioJWBH0PHro1qsfdVJ9Q1rdt1Un1Arttjclhx7dsFxt3I4HrdLJDn+wGoZWU6MNIJUrsRpI3QyIzuyW10UdtuN+eSCK6tURUlqabiFJCs8/Fm6JJxMQVjj3mAIJGnsBVNnM41AjuCqmzmYagRu2251FinjO6KyntksgDAXGsjgdgRkHEjAnpjrrTJn0V9rOBNGudQnazgR4bH3HS3OgVDUUlyt8+Y2AljqqeT5qQCyHz6g/TTy2UZabEMP2MB6cpdiGH7FD3r+7ps207Oftd7OqGK0CjmiivFsh6U5SVgiTwr+WQ5VWQeEhgVC4IOa4tw1McdarYfY/yZ7iWCtA6tew8TPVveSTatr9ko88dG9S9esQ5PDK7rhmUdeJjSMBvLIYZyMaqsezptEMezkaW7a+6qmmUU61xRBkGIuCgwOvhPTzPy+GtFQyWDuGsv6WSwdw1jm2fueWOn9KzJDSRfiPJxSKFMgZdm8KDyGST8BqzranGXm2UehH1NNC67KPQlJ7wveJtm49oP2ZbRqRWwVU0cl0rlz6IrE3JIYiffHMKzPgDwqFyMnWe4txNMoCmn+fs+ZQ8Tz0yB0qvjz5mbqaqqaOZaikqJIJU9143KsPsR1GqOU8mYd+bwgXim4aw/qZ+Tfycn/dGLHX4Jhix1+CYBcr/AHy88RdrxW1gTqonnaQL9gTgftoWYtux1gli25MA1Eif/9k=", layout="wide")

hide_pages(['prompts'])
#Creating the chatbot interface
#st.title("Mobius: LLM-Powered Chatbot")
st.markdown("""
<h1 style='text-align: center; color: teal;'>Mobius: LLM-Powered Chatbot</h1>
<style>
    .katex .base {
        width: 100%;
        display: flex;
        flex-wrap: wrap;
    }
    .stCodeBlock code {
        white-space: break-spaces !important;
        }
</style>
""", unsafe_allow_html=True)

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'citation' not in st.session_state:
    st.session_state['citation'] = []

if 'document' not in st.session_state:
    st.session_state['document'] = None

###Global variables:###
#REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', default='')
os.environ["REPLICATE_API_TOKEN"] = "r8_E0KsK9OwgkvQZFImixANLwT2zza8FuB2PFaK5"
# #Your your (Replicate) models' endpoints:
# REPLICATE_MODEL_ENDPOINT7B = os.environ.get('REPLICATE_MODEL_ENDPOINT7B', default='')
# REPLICATE_MODEL_ENDPOINT13B = os.environ.get('REPLICATE_MODEL_ENDPOINT13B', default='')
# REPLICATE_MODEL_ENDPOINT70B = os.environ.get('REPLICATE_MODEL_ENDPOINT70B', default='')

# #Dropdown menu to select the model edpoint:
# selected_option = st.sidebar.selectbox('Choose a LLaMA2 model:', ['LLaMA2-70B', 'LLaMA2-13B', 'LLaMA2-7B'], key='model')
# if selected_option == 'LLaMA2-7B':
#     st.session_state['llm'] = REPLICATE_MODEL_ENDPOINT7B
# elif selected_option == 'LLaMA2-13B':
#     st.session_state['llm'] = REPLICATE_MODEL_ENDPOINT13B
# else:
#     st.session_state['llm'] = REPLICATE_MODEL_ENDPOINT70B

# Define a function to clear the input text
def clear_input_text():
    global input_text
    input_text = ""

# We will get the user's input by calling the get_text function
def get_text():
    global input_text
    input_text = st.text_input("Ask your Question", key="input", on_change=clear_input_text)
    return input_text

# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

import docx2txt

def parse_docx(file):
    text_content = docx2txt.process(file)
    return text_content

def convert_document_to_dict(document):
    return {
        'page_content': document.page_content,
        'metadata': document.metadata,  # assuming this is already a dictionary
    }

def save_file(file):
    try:
        base_name, extension = os.path.splitext(file.name)
        file_path = os.path.join("doc/", file.name)

        counter = 1
        while os.path.exists(file_path):
            new_file_name = f"{base_name}_{counter}{extension}"
            file_path = os.path.join("doc/", new_file_name)
            counter += 1
        with open(file_path, "wb") as f:
            f.write(file.read())
    except Exception as e:
        print(f"Error: {e}")
        return None

    return str(file_path)

def main():
    create_db()

    with st.container():
        col1, col2, col3 = st.columns((25,50,25))

        with col2:
            user_input = get_text()
            uploaded_file = st.file_uploader("**Upload Your PDF/DOCX/TXT File**", type=['pdf', 'docx', 'txt'])
    st.markdown("""---""")

    if uploaded_file:
        if not st.session_state.document:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            if file_extension == 'pdf':
                doc = parse_pdf(uploaded_file)
                pages = text_to_docs(doc)
            elif file_extension == 'docx':
                doc = parse_docx(uploaded_file)
                pages = text_to_docs(doc)
            elif file_extension == 'txt':
                pages = text_to_docs(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

            file_path = save_file(uploaded_file)
            print(f'File path: {file_path}')

            retriever = chat.embed_document(pages)

            if file_path:
                st.session_state.document = {
                    "pages": pages,
                    "file_path": file_path,
                    "retriever": retriever
                }
            else:
                st.error("File uploading failed. Try again.")

            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.title("Chat")
                with col2:
                    st.title("Citation")
        elif user_input and user_input.strip() != "":

            document = st.session_state.document

            pages = document['pages']
            document_path = document['file_path']
            retriever = document['retriever']
            #output, sources = chat.answer(user_input, pages)
            #output, sources = chat.answer_Faiss(user_input, pages)
            #output, sources = chat.answer_llm_Faiss(user_input, pages)
            output, sources = chat.answer_replicate_Faiss(user_input, retriever)
            # store the output
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
            # converted_sources = [convert_document_to_dict(doc) for doc in sources]
            converted_sources = [doc.page_content for doc in sources]
            st.session_state.citation.append(converted_sources)

            # Log to database
            log_to_database(user_input, output, converted_sources)
            
    with st.container():
        col1, col2 = st.columns(2, gap="large")

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                with col1:
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))
                with col2:
                    for item in st.session_state["citation"][i]:
                        #wrapped_string = textwrap.fill(item, width=50, break_long_words=True)
                        st.info(str(item), icon="ℹ️")

    # if st.session_state['generated']:
    #     for i in range(len(st.session_state['generated'])-1, -1, -1):
    #         message(st.session_state["generated"][i], key=str(i))
    #     message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# Run the app
if __name__ == "__main__":
    main()
