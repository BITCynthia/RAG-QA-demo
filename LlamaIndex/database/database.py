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
    def __init__(self, data_path, chroma_path, chunk_size = 100, chunk_overlap = 10, index_way = "nodes", reset = True):
        """
        Initialize the Database class.

        :param data_path: Path to the data directory.
        :param chroma_path: Path to save/load the Chroma database.
        :param index_way: Method of indexing ('nodes' or 'doc').
        :param save: Whether to save the index.
        """
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if reset:
            self.clear_database()
        documents = self.load_documents()
        vector_storage = self.create_vector_storage()
        embedding_function = self.load_embedding_model()
        if index_way == "nodes":
            nodes = self.split_text_into_nodes(documents)
            index = self.index_by_nodes(nodes, vector_storage, embedding_function)
        else:
            index = self.index_by_doc(documents, vector_storage, embedding_function)
        self.save_to_chroma(vector_storage)
        self.retriever = self.make_retriever(index)

    def clear_database(self):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

    def load_documents(self):
        if Path(self.data_path).is_dir():
            return self.__load_from_directory()

    def __load_from_directory(self):
        loader = SimpleDirectoryReader(self.data_path)
        documents = loader.load_data()
        return documents

    def __load_from_txt(self):
        pass

    def __load_from_pdf(self):
        pass

    def create_vector_storage(self, collection_name = "demo"):
        chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        chroma_collection = chroma_client.create_collection(collection_name)
        return self.__initialize_storage_context(chroma_collection)

    def __initialize_storage_context(self, chroma_collection) -> StorageContext:
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context

    def load_embedding_model(self):
        return OpenAIEmbedding().get_embeddings()

    def split_text_into_nodes(self, documents):
        text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        nodes = text_splitter.get_nodes_from_documents(documents=documents)
        return nodes

    def index_by_doc(self, documents, storage_context, embedding_function):
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            embedding_function=embedding_function
        )
        return index

    def index_by_nodes(self, nodes, storage_context, embedding_function):
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embedding_function
        )
        return index

    def make_retriever(self, index):
        return index.as_retriever()

    def save_to_chroma(self, vector_storage):
        if not os.path.exists(self.chroma_path):
            os.makedirs(self.chroma_path)
        vector_storage.persist()

    def retrieve_text(self, query:str):
        return self.retriever.retrieve(query)

if __name__ == "__main__":
    data_path = os.path.abspath("../data/txt")
    chroma_path = os.path.abspath("chroma")
    logger.note(f"Data path: {data_path}")
    logger.note(f"Chroma path: {chroma_path}")
    db = Database(data_path, chroma_path)