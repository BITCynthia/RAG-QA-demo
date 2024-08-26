import os
import sys
import shutil
from pathlib import Path
from tclogger import logger

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from llama_index.legacy.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex

llama_index_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(llama_index_path)
from agent.embedding import OpenAIEmbedding

class Database():
    def __init__(self, chunk_size=100, chunk_overlap=10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever = None

    def load_from_documents(self, data_path, save_path, reset=True, index_way="nodes", collection_name="demo"):
        if reset:
            self.clear_database(chroma_path)
        documents = self.__load_from_directory(data_path)
        embedding_function = self.load_embedding_model()
        if index_way == "nodes":
            nodes = self.split_text_into_nodes(documents)
            index = self.index_by_nodes(nodes, embedding_function)
        else:
            index = self.index_by_doc(documents, embedding_function)
        self.save_to_disk(index, save_path)
        self.retriever = self.make_retriever(index)

    def load_from_database(self, chroma_path, collection_name="demo"):
        vector_storage = self.load_vector_storage(chroma_path, collection_name)
        embedding_function = self.load_embedding_model()
        index = self.load_index(vector_storage, embedding_function)
        self.retriever = self.make_retriever(index)

    def clear_database(self, chroma_path):
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)

    def __load_from_directory(self, data_path):
        loader = SimpleDirectoryReader(data_path)
        documents = loader.load_data()
        return documents

    def load_embedding_model(self):
        return OpenAIEmbedding().get_embeddings()

    def split_text_into_nodes(self, documents):
        text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        nodes = text_splitter.get_nodes_from_documents(documents=documents)
        return nodes

    def index_by_doc(self, documents, embedding_function):
        index = VectorStoreIndex.from_documents(
            documents=documents,
            embedding_function=embedding_function
        )
        return index

    def index_by_nodes(self, nodes, embedding_function):
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embedding_function
        )
        return index

    def load_index(self, storage_context, embedding_function):
        return VectorStoreIndex(
            storage_context=storage_context,
            embed_model=embedding_function
        )

    def make_retriever(self, index):
        return index.as_retriever()

    def save_to_disk(self, index, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        index.storage_context.persist(save_path)

    def retrieve_text(self, query: str):
        if self.retriever:
            return self.retriever.retrieve(query)
        else:
            raise ValueError("Database not loaded. Please load the database first.")

if __name__ == "__main__":
    # Example usage:
    data_path = os.path.abspath("../data/txt")
    chroma_path = os.path.abspath("chroma")
    logger.note(f"Data path: {data_path}")
    logger.note(f"Chroma path: {chroma_path}")

    db = Database()
    db.load_from_documents(data_path, chroma_path, reset=True, index_way="nodes", collection_name="demo")
    # or
    # db.load_from_database(chroma_path, collection_name="demo")

    query = "example query"
    results = db.retrieve_text(query)
    for result in results:
        print(result)
