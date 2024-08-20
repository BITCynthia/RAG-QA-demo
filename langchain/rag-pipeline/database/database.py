import os
import sys
import shutil
from tclogger import logger

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma

langchain_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(langchain_path)
from agent.embedding import OpenAIEmbedding

class Database():
    def __init__(self, data_path: str, chroma_path: str, reset: bool = False):
        if reset:
            self.clear_database()
        documents = self.load_from_pdf(data_path)
        chunks = self.split_docments(documents)
        database=self.load_database(chroma_path)
        self.updata_database(database, chunks)

    def clear_database(self, chroma_path):
        if os.path.exist(chroma_path):
            shutil.rmtree(chroma_path)

    def load_from_pdf(self, data_path):
        document_loader = PyPDFDirectoryLoader(data_path)
        return document_loader.load()

    def split_docments(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False
        )
        return text_splitter.split_documents(documents)

    def load_database(self, chroma_path):
        embedding_function = OpenAIEmbedding().get_embeddings()
        db = Chroma(
            persist_directory=chroma_path,
            embedding_function=embedding_function
        )
        return db

    def updata_database(self, db, chunks):
        def find_new_chunks():
            new_chunks = []
            for chunk in chunks_with_ids:
                if chunk.metadata["id"] not in existing_ids:
                    new_chunks.append(chunk)
            return new_chunks

        def updata_schema():
            new_chunk_with_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_with_ids)
            db.persist()

        existing_ids = set(db.get(include=[])["ids"])
        logger.note(f"Number of existing documents in DB: {len(existing_ids)}")

        chunks_with_ids = self.calculate_chunk_ids(chunks)
        new_chunks = find_new_chunks()

        if len(new_chunks):
            logger.note(f"Adding new documents: {len(new_chunks)}")
            updata_schema()
        else:
            logger.note(f"No new documents to add.")

    def calculate_chunk_ids(self, chunks):
        # ID format: page_source:page number:chunk index
        last_page_id = None
        current_page_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_page_index += 1
            else:
                current_page_index = 0 # new page

            chunk_id = f"{current_page_id}:{current_page_index}"
            last_page_id = current_page_id

            chunk.metadata["id"] = chunk_id

        return chunks
    
if __name__ == "__main__":
    data_path = os.path.abspath('data/pdf')
    chroma_path = os.path.abspath('langchain/rag-pipeline/chroma')
    print(data_path)
    print(chroma_path)
    Database(data_path, chroma_path, reset = False)