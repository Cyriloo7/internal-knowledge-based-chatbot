# client/src/components/retriever.py

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


class Retriever:
    def __init__(self, collection_name, vectorstore_path="chroma_vectorstore"):
        load_dotenv()
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore_path = vectorstore_path
        self.collection_name = collection_name
        self.vectorstore_path = vectorstore_path

    def get_vector_store(self):
        path = os.path.join(os.getcwd(), self.vectorstore_path, self.collection_name)
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=path,
        )
        return vector_store