from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class Embedder:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def embed_and_store(self, chunks):
        """Embeds the chunks and stores them in a vector store."""
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        return vector_store
