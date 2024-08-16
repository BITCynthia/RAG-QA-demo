import os
import shutil
import logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from agent.embedding import OpenAIEmbedding


def load_docments(DATA_PATH):
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    docments = loader.load()
    return docments

def split_text(docments: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docments)
    print(f"Split {len(docments)} docments into {len(chunks)} chunks.")

    # example_docment = chunks[10]
    # print("examppe: ")
    # print(example_docment.page_content)
    # print(example_docment.metadata)
    
    return chunks

def save_to_chroma(chunks, CHROMA_PATH):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OpenAIEmbedding().get_embeddings()
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Save {len(chunks)} chunks to {CHROMA_PATH}")

def main():
    DATA_PATH = "documents/"
    CHROMA_PATH = "chroma"

    docments = load_docments(DATA_PATH)
    chunks = split_text(docments)
    save_to_chroma(chunks, CHROMA_PATH)

if __name__ == '__main__':
    main()