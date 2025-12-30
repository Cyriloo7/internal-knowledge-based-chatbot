import io
import os
from langchain_core.documents import Document
from dotenv import load_dotenv
from operator import add as add_messages
from typing import TypedDict, Annotated, Sequence

import fitz
from langgraph.graph import StateGraph, END
from PIL import Image

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

import google.generativeai as genai

memory = MemorySaver()

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2
)

vision_model = genai.GenerativeModel('models/gemini-2.5-flash')

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


pdf_path = r"Documents\AI_ML R&D v1.docx.pdf"

if not os.path.exists(pdf_path):
    raise FileExistsError(f"The file {pdf_path} does not exist.")

loader = PyPDFLoader(pdf_path)

try:
    pages = loader.load()
except Exception as e:
    raise RuntimeError(f"Failed to load PDF file: {e}")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
docs_split = text_splitter.split_documents(pages)
for doc in docs_split:
    doc.metadata.update({"type": "text", "source": pdf_path})


# Extract images from documents
def extract_images_from_doc(pdf_path):
    doc = fitz.open(pdf_path)
    images =[]
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            images.append((page_index+1, image_bytes))
    return images

def image_to_document(image_bytes, page_num):
    image = Image.open(io.BytesIO(image_bytes))

    #OCR
    import pytesseract
    ocr_text = pytesseract.image_to_string(image).strip()

    #gemini caption
    caption_prompt = """
    Generate a concise technical caption for this image.
    Describe diagrams, architecture, flow, components, and relationships.
    """
    caption = vision_model.generate_content([caption_prompt, image])
    caption = caption.text.strip()

    combined_text = f"""
    IMAGE CAPTION:
    {caption}
    OCR TEXT:
    {ocr_text}
    """.strip()

    return Document(
        page_content=combined_text,
        metadata={
            "type": "image",
            "source": pdf_path,
            "page_num": page_num,
        }
    )


image_doc = []
for pagenum, image_bytes in extract_images_from_doc(pdf_path):
    try:
        image_doc.append(image_to_document(image_bytes, pagenum))
    except Exception as e:
        print(f"Error processing image on page {pagenum}: {e}")


vectorstorepath = "chroma_vectorstore"
collection_name = "multi_modal_docs"

path = os.path.join(os.getcwd(), vectorstorepath, collection_name)
os.makedirs(path, exist_ok=True)

import hashlib

def doc_id(doc):
    content = doc.page_content
    source = doc.metadata.get("source", "")
    h = hashlib.sha256(f"{source}:{content}".encode()).hexdigest()
    return h

all_docs = docs_split + image_doc
ids = [doc_id(doc) for doc in all_docs]

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=path,
)
existing = vector_store._collection.get(include=[])
existing_ids = set(existing["ids"])
# print(existing_ids)
# exit()
new_docs = []
new_ids = []

for doc, id_ in zip(all_docs, ids):
    if id_ not in existing_ids:
        new_docs.append(doc)
        new_ids.append(id_)


try:
    if new_docs:
        vector_store.add_documents(new_docs, ids=new_ids)
        # vector_store.persist()
    else:
        print("✅ No new documents to add")

    # vectorstore = Chroma.from_documents(
    #     documents=docs_split, 
    #     embedding=embeddings,
    #     persist_directory=path,
    #     collection_name="my_collection"
    # )

except Exception as e:
    raise RuntimeError(f"Failed to create or persist Chroma vector store: {e}")


retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@tool
def document_search_tool(query: str) -> str:
    """Tool to search documents."""
    results = retriver.invoke(query)
    combined_text = "\n".join([f"{doc.page_content}, {str(doc.metadata)}" for doc in results])
    return combined_text

 
tools =[document_search_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """check if the last message contains tool calls"""
    result = state['messages'][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt = """You are an intelligent document-aware AI assistant.

Your role is to answer user questions by retrieving and synthesizing information from the provided document knowledge base, which contains:
- Text extracted from PDF pages
- Images extracted from the same PDFs, represented using:
  - Technical captions generated by a vision model
  - OCR-extracted text from the images

You have access to the following tool:
{tools}

### Tool Usage Rules
- When a user query requires information from the documents, you MUST call the document_search_tool.
- A query requires document search if it asks about:
  - Specific facts, figures, diagrams, tables, or processes
  - Content that could plausibly exist in the provided PDFs
  - Anything referencing "this document", "the manual", or "the report"
- Formulate a concise and meaningful search query when calling the tool.
- You may call the tool multiple times if necessary.
- Do NOT infer, assume, or fabricate information that is not explicitly supported by the retrieved documents.

### Answering Rules
- Synthesize information from all relevant retrieved results (text + image-derived content).
- Prefer factual, technical, and structured explanations.
- If only part of the answer is found, answer only that part and clearly state what is missing.
- If information is insufficient or not found in the documents, clearly state that.
- If multiple sources conflict, report the discrepancy and cite all relevant pages.

### Citations & References (MANDATORY)
At the end of every answer that uses document information:
- List the document source name
- Mention the page number(s) referenced
- Clearly indicate whether the information came from:
  - Text content
  - Image caption
  - Image OCR

### Response Format
- Provide a clear and complete answer first.
- Add a **References** section at the end in the following format:

References:
- Document: <document name>
  Page: <page number>
  Type: Text / Image Caption / Image OCR

### General Behavior
- Be concise but thorough.
- Use cautious language when information is ambiguous.
- Do not expose internal tool execution details.
- Do not mention embeddings, vector stores, or implementation details.
- If the question does not require document lookup, answer using general knowledge without calling tools.
"""

tools_dict  = {our_tool.name: our_tool for our_tool in tools}


#LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Call the LLM with the current state messages"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)]+messages
    message = llm.invoke(messages)
    return {"messages": [message]}



def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from LLM's response"""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        # print(t)
        # break
        if not t['name'] in tools_dict:
            print(f"Tool {t['name']} not found")
        
        tool_output  = tools_dict[t['name']].invoke(t['args'])

        results.append(ToolMessage(tool_call_id = t['id'], name=t['name'], content=str(tool_output)))
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriver_agent", take_action)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriver_agent", False: END}
)
graph.add_edge("retriver_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id":"1"}}

def running_agent():
    print(" RAG Agent is running... ")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting RAG Agent.")
            break

        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages}, config=config)
        print("\n======ANSWER======")
        print(result['messages'][-1].text)

running_agent()