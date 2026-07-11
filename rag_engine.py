import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
from dotenv import load_dotenv

load_dotenv()

class MedicalRAGEngine:
    def __init__(self, data_path="data/sample_reports.json", persist_directory="./chroma_db"):
        self.data_path = data_path
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        
    def load_and_index(self):
        """Load reports from JSON and index them into ChromaDB."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        with open(self.data_path, 'r') as f:
            reports = json.load(f)
            
        documents = []
        for item in reports:
            content = f"Type: {item['type']}\nReport: {item['report']}\nHistory: {item['patient_history']}"
            metadata = {"id": item['id'], "type": item['type']}
            documents.append(Document(page_content=content, metadata=metadata))
            
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"Indexed {len(splits)} chunks from {len(reports)} reports.")

    def get_retriever(self):
        """Returns the retriever object."""
        if self.vector_store is None:
            # Try to load from disk
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self.vector_store.as_retriever(search_kwargs={"k": 3})

    def search_reports(self, query):
        """Search for relevant report chunks."""
        retriever = self.get_retriever()
        return retriever.get_relevant_documents(query)

if __name__ == "__main__":
    # Test indexing
    engine = MedicalRAGEngine()
    engine.load_and_index()
    results = engine.search_reports("pneumonia")
    for res in results:
        print(f"Match: {res.metadata['id']} - {res.page_content[:100]}...")
