import os
import tempfile
import requests
from typing import List
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv() 

# --- Configuration ---
API_TOKEN = "3b3b7f8e0cb19ee38fcc3d4874a8df6dadcdbfec21b7bbe39a73407e2a7af8a0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Authentication ---
auth_scheme = HTTPBearer()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx 6.0 Intelligent Query-Retrieval System",
    description="An API that processes a document URL and answers questions about it."
)

# --- Load Models (once on startup) ---
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print("Initializing Google Gemini model...")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, convert_system_message_to_human=True)
print("Models loaded successfully.")


# --- Pydantic Models for API ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str
    source_documents: List[dict]

class QueryResponse(BaseModel):
    answers: List[Answer]

# --- Token Verification Dependency ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing authentication token")
    return credentials

# --- API Endpoints ---
@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_query(request: QueryRequest):
    document_url = request.documents
    questions = request.questions
    tmp_file_path = None

    try:
        # 1. Download the PDF from the URL
        print(f"Downloading document from: {document_url}")
        response = requests.get(document_url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        
        # 2. Load and process the downloaded PDF
        print(f"Loading downloaded PDF: {tmp_file_path}")
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        # 3. Create an in-memory FAISS index
        print("Creating in-memory FAISS index...")
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # 4. Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True
        )

        # 5. Answer questions
        processed_answers = []
        for question in questions:
            print(f"Answering question: '{question}'")
            result = qa_chain.invoke({"query": question})
            sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in result.get('source_documents', [])]
            processed_answers.append(
                Answer(question=question, answer=result.get("result", "No answer found"), source_documents=sources)
            )
        
        return QueryResponse(answers=processed_answers)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    finally:
        # 6. Clean up the temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            print(f"Cleaned up temporary file: {tmp_file_path}")


@app.get("/")
def read_root():
    return {"message": "HackRx 6.0 FAISS & Gemini solution is running. Go to /docs to test the API."}
