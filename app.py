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

st.set_page_config(layout="wide")
#Creating the chatbot interface
#st.title("Mobius: LLM-Powered Chatbot")
st.markdown("<h1 style='text-align: center; color: black;'>Mobius: LLM-Powered Chatbot</h1>", unsafe_allow_html=True)

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'citation' not in st.session_state:
    st.session_state['citation'] = []

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
            chunk_size=2000,
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

def main():

    with st.container():
        col1, col2, col3 = st.columns((25,50,25))

        with col2:
            user_input = get_text()
            uploaded_file = st.file_uploader("**Upload Your PDF/DOCX/TXT File**", type=['pdf', 'docx', 'txt'])
    st.markdown("""---""")

    if user_input:
        if uploaded_file:
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

            output, sources = chat.answer(user_input, pages)
            # store the output
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
         #   converted_sources = [convert_document_to_dict(doc) for doc in sources]
            converted_sources = [doc.page_content for doc in sources]
            st.session_state.citation.append(converted_sources)

            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.title("Chat")
                with col2:
                    st.title("Citation")


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
