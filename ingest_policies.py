from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
import os

# Load environment variables (for HF_TOKEN if set)
load_dotenv()

path = "C:/Users/priya/OneDrive/Desktop/acko/DataSet/Policy_Documents/Acko_Insurance_FAQs.pdf"

def ingest_pdfs(pdf_paths):
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Hugging Face embeddings (token picked up automatically if set)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Persist ChromaDB automatically when persist_directory is set
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("✅ PDFs ingested into ChromaDB")

if __name__ == "__main__":
    ingest_pdfs([path])
