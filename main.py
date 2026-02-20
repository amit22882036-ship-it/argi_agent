import os
import glob
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_openai import ChatOpenAI

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
# -----------------------

from weather_service import WeatherService

load_dotenv()

app = FastAPI()

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


# --- 1. Initialize Services ---
weather_service = WeatherService()

# --- 2. Initialize Embeddings ---
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# --- 3. Setup Local Vector Store ---
LOCAL_INDEX_FOLDER = "local_faiss_index"


def load_or_build_index():
    if os.path.exists(LOCAL_INDEX_FOLDER):
        print(f"✅ Found local index in '{LOCAL_INDEX_FOLDER}'. Loading...")
        try:
            return FAISS.load_local(LOCAL_INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ Error loading index: {e}. Rebuilding...")

    print(f"⚠️ Building index from 'project_sources/' PDFs...")
    pdf_files = glob.glob("project_sources/*.pdf")

    if not pdf_files:
        print("❌ ERROR: No PDF files found in project_sources/!")
        return None

    print(f"📄 Found {len(pdf_files)} PDF files to process...")
    documents = []

    for pdf_path in pdf_files:
        try:
            print(f"   Processing: {os.path.basename(pdf_path)}...")
            doc = fitz.open(pdf_path)

            page_count = len(doc)
            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text()

                # Skip empty pages
                if not text.strip():
                    continue

                # Create document with metadata
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "total_pages": page_count
                    }
                ))

            doc.close()
            print(f"   ✅ Extracted {page_count} pages from {os.path.basename(pdf_path)}")

        except Exception as e:
            print(f"   ⚠️ Error processing {pdf_path}: {e}")
            continue

    if not documents:
        print("❌ No documents extracted from PDFs.")
        return None

    print(f"📝 Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks from PDFs")

    if not chunks:
        print("❌ No text chunks created.")
        return None

    print("🔨 Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(LOCAL_INDEX_FOLDER)
    print("✅ Index built and saved locally!")
    return vectorstore


vectorstore = load_or_build_index()


# --- 4. Initialize LLM (FIXED PARAMETERS) ---
def get_llm_model():
    api_key = os.getenv("LLMOD_API_KEY", "")
    base_url = os.getenv("LLMOD_API_BASE", "https://api.llmod.ai/v1")

    if api_key:
        print(f"🔄 Attempting to connect to llmod.ai...")
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model="RPRTHPB-gpt-5-mini",
                # --- תיקון קריטי: שינוי הטמפרטורה ל-1 לפי דרישת השגיאה ---
                temperature=1
            )
            llm.invoke("test connection")
            print("✅ Connected successfully to llmod.ai")
            return llm
        except Exception as e:
            print(f"\n❌ ERROR Connecting to llmod.ai: {e}\n")

    print("⚠️ Switching to BACKUP (Local Ollama)...")
    return ChatOllama(model="llama3", temperature=1, keep_alive="5m")


llm = get_llm_model()

# --- 5. Manual Logic Setup ---

# Global Memory Store
chat_sessions = {}


def get_history(session_id: str):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatMessageHistory()
    return chat_sessions[session_id]


# Prompts
rephrase_prompt = ChatPromptTemplate.from_template(
    """Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.

    Chat History:
    {history}

    Follow Up Input: {question}

    Standalone Question:"""
)

advisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent agricultural advisor for Israeli farmers.
    Use the provided Context and Weather Data to answer the user's question.

    Weather Conditions: 
    {weather}

    Relevant Book Excerpts:
    {context}

    Instructions:
    1. Base your answer primarily on the context and weather.
    2. If the context doesn't contain the answer, say you don't know.
    3. Prioritize organic methods if mentioned.
    """),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])


# --- 6. API Endpoint ---
class AdviceRequest(BaseModel):
    session_id: str
    city: str
    query: str
    date: str | None = None


@app.post("/get-advice")
async def get_farming_advice(request: AdviceRequest):
    try:
        if not vectorstore:
            raise HTTPException(status_code=500, detail="Vector store not initialized.")

        # 1. Get History & Weather
        history_obj = get_history(request.session_id)
        current_history = history_obj.messages
        weather_info = weather_service.get_weather(request.city, request.date)

        print(f"Thinking... Session: {request.session_id} | City: {request.city}")

        # 2. Rephrase Question (if history exists)
        search_query = request.query
        if current_history:
            hist_str = "\n".join([f"{m.type}: {m.content}" for m in current_history[-4:]])

            chain = rephrase_prompt | llm | StrOutputParser()
            try:
                search_query = chain.invoke({"history": hist_str, "question": request.query})
                print(f"🔄 Rephrased: {search_query}")
            except Exception as e:
                print(f"⚠️ Rephrase failed ({e}), using original query.")

        # 3. Retrieve Documents
        docs = vectorstore.similarity_search(search_query, k=5)
        context_text = "\n\n".join([d.page_content for d in docs])

        # 4. Generate Answer
        final_chain = advisor_prompt | llm | StrOutputParser()

        response = final_chain.invoke({
            "weather": weather_info,
            "context": context_text,
            "history": current_history[-6:],
            "question": request.query
        })

        # 5. Update Memory
        history_obj.add_user_message(request.query)
        history_obj.add_ai_message(response)

        return {
            "advice": response,
            "weather_context_used": weather_info,
            "model_used": "LLMod/Manual"
        }

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
