import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import io, requests


chain = load_qa_chain(OpenAI(), chain_type="stuff")
embeddings = HuggingFaceEmbeddings()

@st.cache_resource()
def create_vector_db(texts, _embeddings):
    return FAISS.from_texts(texts, _embeddings)

def pdf_to_raw_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        raw_text = ''
        with io.BytesIO(response.content) as open_pdf_file:
            reader = PdfReader(open_pdf_file)
            for _, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
            num_pages = len(reader.pages)
            print(num_pages)
        return raw_text
    else:
        st.write(f"Error {response.status_code}: Unable to download PDF '{url}'")
        return False
# Replace 'output_dir' with the path to the directory where you want to save the PDF
output_dir = "."


# Streamlit app
st.title("Talk with your favorite pdf!")

# get gender from user
#gender = st.selectbox("Choose an option:", ['male','female'])


pdf_url = st.text_input("Enter the URL for a pdf (e.g. https://arxiv.org/pdf/2303.17564.pdf):")
if len(pdf_url) != 0:
    st.write("## The first time this runs for a new pdf it may take up to a minute to load the model")
    raw_text = pdf_to_raw_text(pdf_url)

    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    docsearch = create_vector_db(texts, embeddings)

    query = st.text_input("Enter your question (e.g. who are the authors of this paper?):")
    if len(query) != 0:
        docs = docsearch.similarity_search(query)
        st.write(chain.run(input_documents=docs, question=query))
