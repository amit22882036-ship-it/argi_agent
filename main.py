import os
import glob
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_openai import ChatOpenAI
# Use the updated import if you have langchain-ollama installed, otherwise fallback
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Local Vector Store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# -----------------------

from weather_service import WeatherService

load_dotenv()

app = FastAPI()

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


# 1. Initialize Services
weather_service = WeatherService()

# 2. Initialize Embeddings (Local GPU/MPS)
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'mps'}  # Use 'cpu' if MPS fails on your machine
)

# 3. Setup Local Vector Store (Replaces Pinecone)
LOCAL_INDEX_FOLDER = "local_faiss_index"


def load_or_build_index():
    """
    Checks if a local index exists.
    If yes: Loads it.
    If no: Reads output_data/, builds index, and saves it.
    """
    if os.path.exists(LOCAL_INDEX_FOLDER):
        print(f"✅ Found local index in '{LOCAL_INDEX_FOLDER}'. Loading...")
        return FAISS.load_local(LOCAL_INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

    print(f"⚠️ No local index found. Building from 'output_data/'...")

    # --- Ingestion Logic ---
    all_texts = []
    files = glob.glob("output_data/*.json")

    if not files:
        print("❌ ERROR: No JSON files found in output_data/!")
        return None

    for filepath in files:
        print(f"Processing {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Recursive extractor for your JSON structure
        def extract_text(node, current_path=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    new_path = f"{current_path} > {key}" if current_path else key
                    extract_text(value, new_path)
            elif isinstance(node, str):
                content = f"Source: {os.path.basename(filepath)}\nSection: {current_path}\nContent: {node}"
                all_texts.append(content)

        extract_text(data)

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents(all_texts)
    print(f"Generated {len(chunks)} chunks. Embedding now (this may take a minute)...")

    # Build and Save FAISS Index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(LOCAL_INDEX_FOLDER)
    print("✅ Index built and saved locally!")
    return vectorstore


# Initialize Vectorstore on Startup
vectorstore = load_or_build_index()


# 4. Initialize LLM
def get_llm_model():
    """
    Attempts to use llmod.ai (Primary).
    Falls back to Ollama/Llama3 (Backup) if key is missing or connection fails.
    """
    api_key = os.getenv("LLMOD_API_KEY")
    base_url = os.getenv("LLMOD_API_BASE", "https://api.llmod.ai/v1")

    if api_key:
        try:
            print(f"Connecting to llmod.ai...")
            # Using specific model from your previous logs
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model="RPRTHPB-gpt-5-mini",
                temperature=1  # UPDATED: Set to 1 as requested
            )
            # Simple test to verify connection
            llm.invoke("test")
            print("✅ Connected to llmod.ai")
            return llm
        except Exception as e:
            print(f"⚠️ Connection failed ({e}). Switching to backup...")

    print("✅ USING BACKUP: Local Ollama (Llama3)")
    return ChatOllama(
        model="llama3",
        temperature=1,  # UPDATED: Set to 1 as requested
        keep_alive="5m"
    )


llm = get_llm_model()


# 5. Define Prompt
prompt_template = """
You are an intelligent agricultural advisor for Israeli farmers. 
Your goal is to give practical advice using the specific book excerpts provided and the weather data.

Situation:
User Question: "{question}"
Weather Context: 
{weather_info}

Book Excerpts:
{context}

Instructions:
1. Analyze the Book Excerpts for relevant advice.
2. Consider the specific Weather conditions (Rain, Wind, Temp).
3. Combine them to give actionable advice.
4. If "Organic" methods are mentioned, prioritize them.

Your Advice, respond in Englis:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "weather_info", "question"]
)


# 6. Request Model
class AdviceRequest(BaseModel):
    city: str
    query: str
    date: str | None = None


# 7. API Endpoint
@app.post("/get-advice")
async def get_farming_advice(request: AdviceRequest):
    try:
        if not vectorstore:
            raise HTTPException(status_code=500, detail="Vector store not initialized. Check server logs.")

        # Step A: Get Weather
        weather_info = weather_service.get_weather(request.city, request.date)

        # Step B: Local Retrieval
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        retrieved_docs = retriever.invoke(request.query)

        # Debug Print
        print(f"--- Retrieved {len(retrieved_docs)} chunks ---")

        # Step C: Run Chain
        def format_docs(docs):
            return "\n\n".join(f"[From {d.metadata.get('source', 'Unknown')}]: {d.page_content}" for d in docs)

        rag_chain = (
                {
                    "context": lambda x: format_docs(retrieved_docs),
                    "question": RunnablePassthrough(),
                    "weather_info": lambda x: weather_info
                }
                | prompt
                | llm
                | StrOutputParser()
        )

        model_name = "llmod.ai" if isinstance(llm, ChatOpenAI) else "Local Ollama"
        print(f"Thinking... (Using {model_name})")
        result = rag_chain.invoke(request.query)

        return {
            "advice": result,
            "weather_context_used": weather_info,
            "model_used": model_name
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)