# client/src/components/indexer.py

import io
import os
from PIL import Image
from langchain_core.documents import Document
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
from langchain_core.documents import Document


class Indexer:
    def __init__(self, pdf_path):
        load_dotenv()
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.pdf_path = pdf_path
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vision_model = genai.GenerativeModel('models/gemini-2.5-flash')
        # self.vector_store = Chroma(collection_name="documents", embedding_function=self.embeddings, persist_directory="chroma_db")

    def extract_images_from_doc(self, pdf_path):
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
    
    def image_to_document(self, image_bytes, page_num):
        image = Image.open(io.BytesIO(image_bytes))

        #OCR
        import pytesseract
        ocr_text = pytesseract.image_to_string(image).strip()

        #gemini caption
        caption_prompt = """
        Generate a concise technical caption for this image.
        Describe diagrams, architecture, flow, components, and relationships.
        """
        caption = self.vision_model.generate_content([caption_prompt, image])
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
                "source": self.pdf_path,
                "page_num": page_num,
            }
        )
    
    def index_document(self, level):
        if not os.path.exists(self.pdf_path):
            raise FileExistsError(f"The file {self.pdf_path} does not exist.")

        loader = PyPDFLoader(self.pdf_path)

        try:
            pages = loader.load()
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF file: {e}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        docs_split = text_splitter.split_documents(pages)
        for doc in docs_split:
            doc.metadata.update({"type": "text", "source": self.pdf_path})
    

        
        image_doc = []
        for pagenum, image_bytes in self.extract_images_from_doc(self.pdf_path):
            try:
                image_doc.append(self.image_to_document(image_bytes, pagenum))
            except Exception as e:
                print(f"Error processing image on page {pagenum}: {e}")


        vectorstorepath = "chroma_vectorstore"
        collection_name = level

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
            embedding_function=self.embeddings,
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
                print(f"✅ Indexed {len(new_docs)} new documents into collection '{collection_name}'")
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
        

# if __name__ == "__main__":
#     level = "level1"
#     path = input("Enter the path to the PDF file: ")
#     indexer = Indexer(path)
#     indexer.index_document(level=level)