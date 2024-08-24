import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import GPT2TokenizerFast


class PDFLoader:
    def __init__(self, chunk_size=512, chunk_overlap=24):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def load_and_chunk_pdfs(self, pdf_dir):
        """Loads PDFs from the directory and chunks the text."""
        documents = []

        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_dir, filename)
                loader = PyPDFLoader(filepath)
                pages = loader.load_and_split()
                documents.extend(pages)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.count_tokens,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
