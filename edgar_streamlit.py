import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import chat
import streamlit as st
from streamlit_chat import message
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Iterator
from langchain.schema import Document

from pypdf import PdfReader

import config

import os
import json
from pathlib import Path
import pdfkit
import fpdf
from fpdf import FPDF
#import weasyprint

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
    input_text = st.text_input("Input Company Ticker", key="input1", on_change=clear_input_text)
    return input_text


# Define a function to clear the input text
def clear_input_question():
    global input_question
    input_question = ""

# We will get the user's input by calling the get_text function
def get_question():
    global input_question
    input_question = st.text_input("Input Your Question", key="input2", on_change=clear_input_question)
    return input_question

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


def main():
    with open("/Users/xiang/PycharmProjects/Context-Based-LLMChatbot/kaggle/company_tickers_exchange.json", "r") as f:
        CIK_dict = json.load(f)

    # convert CIK_dict to pandas
    CIK_df = pd.DataFrame(CIK_dict["data"], columns=CIK_dict["fields"])


    #user_input = get_text()
    user_question = get_question()

    col_one_list = CIK_df["ticker"].tolist()

    unique_sorted = sorted(list(set(col_one_list)))

    user_select = st.selectbox('Select company ticker', unique_sorted)

    #uploaded_file = st.file_uploader("**Upload Your PDF/DOCX/TXT File**", type=['pdf', 'docx', 'txt'])
    st.markdown("""---""")
    

    if user_select:
#        print(CIK_df[:10])

        CIK = CIK_df[CIK_df["ticker"] == user_select].cik.values[0]

                # preparation of input data, using ticker and CIK set earlier
        url = f"https://data.sec.gov/submissions/CIK{str(CIK).zfill(10)}.json"

        # read response from REST API with `requests` library and format it as python dict
        import requests
        header = {
        "User-Agent": "your.email@email.com"#, # remaining fields are optional
        #    "Accept-Encoding": "gzip, deflate",
        #    "Host": "data.sec.gov"
        }

        company_filings = requests.get(url, headers=header).json()

        company_filings_df = pd.DataFrame(company_filings["filings"]["recent"])
        print(company_filings_df[:10])

        access_number = company_filings_df[company_filings_df.form == "10-K"].accessionNumber.values[0].replace("-", "")

        file_name = company_filings_df[company_filings_df.form == "10-K"].primaryDocument.values[0]

        url_file = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{access_number}/{file_name}"
        print(f"url_file is {url_file}")

        # dowloading and saving requested document to working directory
        req_content = requests.get(url_file, headers=header).content.decode("utf-8")

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(req_content, 'html.parser')
        req_content_text = soup.get_text()

        print(f"len of req_content is {len(req_content)}") 
        # with open(file_name, "w") as f:
        #     f.write(req_content)
        # import pdfkit
        # pdfkit.from_url(file_name, 'out1.pdf')

        if user_question:
            pages = text_to_docs(req_content_text)
            output, sources = chat.answer_Faiss_rate(user_question, pages)

            st.session_state.past.append(user_question)
            st.session_state.generated.append(output)
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
        #print("session is: ", st.session_state)
        required_keys = ['generated', 'past', 'citation']

        if all(st.session_state.get(key) for key in required_keys):

            for i in range(len(st.session_state['generated'])-1, -1, -1):
                #app_state = json.dumps(st.session_state._state.to_dict())
                with col1:
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))
            with col2:
                #  item_list = []
                for item in st.session_state["citation"][-1]:
                    st.info(str(item), icon="ℹ️")




# Run the app
if __name__ == "__main__":
    main()