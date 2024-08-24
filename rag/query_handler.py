from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import VectorStore


class QueryHandler:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.chat_history = []

    def handle_query(self, query_text):
        """Handles the query by retrieving documents from the vector store and
        returns an augmented response."""
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

        retriever = self.vector_store.as_retriever()
        docs = retriever.get_relevant_documents(query_text)

        self.chat_history.append(query_text)

        result = combine_docs_chain.run({
            "input_documents": docs,
            "question": query_text
        })

        self.chat_history.append(result)

        return result
