import os
import io
import hashlib
from dotenv import load_dotenv
from operator import add as add_messages
from typing import TypedDict, Annotated, Sequence

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

import google.generativeai as genai


# ------------------------------------------------------------------
# ENV
# ------------------------------------------------------------------

load_dotenv()
memory = MemorySaver()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ------------------------------------------------------------------
# MODELS
# ------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2
)

vision_model = genai.GenerativeModel('models/gemini-2.5-flash')

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# ------------------------------------------------------------------
# PDF PATH
# ------------------------------------------------------------------

pdf_path = r"Documents\AI_ML R&D - Phase 2.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(pdf_path)

# ------------------------------------------------------------------
# 1️⃣ TEXT EXTRACTION
# ------------------------------------------------------------------

loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=500
)

text_docs = text_splitter.split_documents(pages)

for d in text_docs:
    d.metadata.update({
        "type": "text",
        "source": pdf_path,
    })

# ------------------------------------------------------------------
# 2️⃣ IMAGE EXTRACTION + GEMINI CAPTION + OCR
# ------------------------------------------------------------------

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            images.append((page_index, image_bytes))

    print(f"🖼️ Extracted {len(images)} images")
    return images


def gemini_caption_image(image: Image.Image) -> str:
    prompt = """
    Generate a concise technical caption for this image.
    Describe diagrams, architecture, flow, components, and relationships.
    """

    response = vision_model.generate_content([prompt, image])
    return response.text.strip()


def image_to_document(image_bytes, page_num):
    image = Image.open(io.BytesIO(image_bytes))

    # OCR (backup signal)
    ocr_text = pytesseract.image_to_string(image).strip()
    # print(f"OCR text extracted: {ocr_text}")

    # Gemini caption (primary signal)
    caption = gemini_caption_image(image)
    # print(f"Gemini caption generated: {caption}")

    combined_text = f"""
    IMAGE CAPTION:
    {caption}

    OCR TEXT:
    {ocr_text}
    """

    return Document(
        page_content=combined_text.strip(),
        metadata={
            "type": "image",
            "page": page_num,
            "source": pdf_path,
        },
    )


image_docs = []

for page_num, image_bytes in extract_images_from_pdf(pdf_path):
    try:
        image_docs.append(
            image_to_document(image_bytes, page_num)
        )
        print(f"✅ Processed image on page {page_num}")
    except Exception as e:
        print(f"❌ Image failed on page {page_num}: {e}")

# ------------------------------------------------------------------
# 3️⃣ VECTOR STORE (TEXT + IMAGE)
# ------------------------------------------------------------------

def doc_id(doc: Document):
    return hashlib.sha256(
        f"{doc.metadata.get('source')}:{doc.metadata.get('page')}:{doc.page_content}".encode()
    ).hexdigest()


all_docs = text_docs + image_docs
all_ids = [doc_id(d) for d in all_docs]

collection_name="multimodal_rag"
vectorstore_path = os.path.join(os.getcwd(), "chroma_vectorstore", collection_name)
os.makedirs(vectorstore_path, exist_ok=True)

vector_store = Chroma(
    collection_name="multimodal_rag",
    embedding_function=embeddings,
    persist_directory=vectorstore_path,
)

existing = vector_store._collection.get(include=[])
existing_ids = set(existing["ids"])

new_docs, new_ids = [], []

for doc, id_ in zip(all_docs, all_ids):
    if id_ not in existing_ids:
        new_docs.append(doc)
        new_ids.append(id_)

if new_docs:
    vector_store.add_documents(new_docs, ids=new_ids)
    # vector_store.persist()
    print(f"✅ Added {len(new_docs)} new chunks")
else:
    print("✅ Vector store already up to date")

# ------------------------------------------------------------------
# 4️⃣ RETRIEVER TOOL
# ------------------------------------------------------------------

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

@tool
def document_search_tool(query: str) -> str:
    """Search both text and image-derived content."""
    docs = retriever.invoke(query)
    return "\n\n".join(
        f"[{d.metadata['type']} | page {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )

tools = [document_search_tool]
llm = llm.bind_tools(tools)

# ------------------------------------------------------------------
# 5️⃣ LANGGRAPH AGENT
# ------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    last = state["messages"][-1]
    return hasattr(last, "tool_calls") and len(last.tool_calls) > 0


system_prompt = """
You are a document RAG assistant.
You can search BOTH text content and image-derived content.
Use the search tool when document lookup is needed.
"""


def call_llm(state: AgentState):
    msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(msgs)
    return {"messages": [response]}


def take_action(state: AgentState):
    outputs = []
    for call in state["messages"][-1].tool_calls:
        tool_fn = document_search_tool
        result = tool_fn.invoke(call["args"])
        outputs.append(
            ToolMessage(
                tool_call_id=call["id"],
                name=call["name"],
                content=result,
            )
        )
    return {"messages": outputs}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever", False: END})
graph.add_edge("retriever", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile(checkpointer=memory)

# ------------------------------------------------------------------
# 6️⃣ RUN LOOP
# ------------------------------------------------------------------

config = {"configurable": {"thread_id": "1"}}

print("✅ Multimodal RAG Agent Running (Text + Images)")

while True:
    q = input("\nUser: ")
    if q.lower() in {"exit", "quit"}:
        break

    result = rag_agent.invoke(
        {"messages": [HumanMessage(content=q)]},
        config=config,
    )

    print("\n====== ANSWER ======")
    print(result["messages"][-1].text)
