# app.py

from client.src.components.indexer import Indexer
from client.src.components.retriever import Retriever
from client.src.components.graph import RAG_Agent



global_level = "level1"  # Default level
level = input("Enter the indexing level (e.g., 'l1', 'l2): ")
global_level = level if level else global_level
indexing = input("Start indexing? (y/n): ")
if indexing.lower() == 'y':
    path = input("Enter the path to the PDF file: ")
    indexer = Indexer(path)
    indexer.index_document(level=level)


retriver = Retriever(collection_name=global_level)
vectorstore = retriver.get_vector_store()

agent = RAG_Agent(vector_store=vectorstore)
agent.running_agent()