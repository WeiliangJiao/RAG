from pdf_loader import PDFLoader
from embedder import Embedder
from vector_store_manager import VectorStoreManager
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate


class Commands:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.vector_store = None

    def init(self, pdf_dir):
        """Initialize the RAG system: load PDFs, chunk, embed, and store in vector store."""
        pdf_loader = PDFLoader()
        chunks = pdf_loader.load_and_chunk_pdfs(pdf_dir)
        embedder = Embedder()
        self.vector_store = embedder.embed_and_store(chunks)
        self.vector_store_manager.save_vector_store(self.vector_store)
        print("Initialization complete. PDFs loaded, chunked, embedded, and stored on disk.")

    def query(self, query_text):
        """Send a query and get an augmented response."""
        if self.vector_store is None:
            self.vector_store = self.vector_store_manager.load_vector_store()

        if self.vector_store is None:
            print("Vector store is empty. Please run 'init' first.")
            return

        llm = OpenAI(temperature=0)

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Given the following context: {context}, answer the question: {question}"
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)

        combine_docs_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )

        docs = self.vector_store.as_retriever().get_relevant_documents(query_text)
        result = combine_docs_chain.run({
            "input_documents": docs,
            "question": query_text
        })

        return result

    def restore(self):
        """Clear the vector store."""
        self.vector_store = None
        self.vector_store_manager.clear_vector_store()
        print("Vector store cleared.")
