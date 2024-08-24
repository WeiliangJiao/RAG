import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


class VectorStoreManager:
    def __init__(self, path='vector_store.faiss'):
        self.path = path

    def clear_vector_store(self):
        """Clears the vector store by returning None."""
        return None

    def save_vector_store(self, vector_store):
        """Saves the vector store to disk."""
        if vector_store is not None:
            vector_store.save_local(self.path)
            return True
        return False

    def load_vector_store(self):
        """Loads the vector store from disk."""
        embeddings = OpenAIEmbeddings()  # Initialize the embeddings
        if os.path.exists(self.path):
            vector_store = FAISS.load_local(
                self.path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        return None
