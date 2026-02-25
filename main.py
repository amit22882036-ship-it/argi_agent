import os
import sqlite3
import glob
import fitz
import json
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- LangChain & Agent Imports ---
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage

# --- RAG Imports ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from weather_service import WeatherService

load_dotenv()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# הגדרת לוגים לטרמינל כדי שתוכל להראות למרצה את ה-MultiQuery בזמן אמת!
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# --- 1. Database Setup ---
DB_PATH = "agri_advisor.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('CREATE TABLE IF NOT EXISTS chats (chat_id TEXT PRIMARY KEY, user_name TEXT, title TEXT)')
    conn.execute(
        'CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, content TEXT)')
    conn.commit()
    conn.close()


init_db()

current_active_date = "2025-01-01"

# --- 2. LLM Setup ---
# הזזנו את ה-LLM לכאן כדי שנוכל להשתמש בו במנוע החיפוש החכם
llm = ChatOpenAI(
    api_key=os.getenv("LLMOD_API_KEY"),
    base_url=os.getenv("LLMOD_API_BASE", "https://api.llmod.ai/v1"),
    model="RPRTHPB-gpt-5-mini",
    temperature=1,
    streaming=True
)

# --- 3. Advanced RAG Setup (Multi-Query Retriever) ---
weather_service = WeatherService()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
LOCAL_INDEX_FOLDER = "local_faiss_index"


def load_index():
    if not os.path.exists("project_sources"):
        os.makedirs("project_sources")
    if os.path.exists(LOCAL_INDEX_FOLDER):
        return FAISS.load_local(LOCAL_INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)
    pdf_files = glob.glob("project_sources/*.pdf")
    if not pdf_files: return None
    docs_list = []
    for pdf in pdf_files:
        doc = fitz.open(pdf)
        for page in doc:
            text = page.get_text()
            if text.strip(): docs_list.append(Document(page_content=text, metadata={"source": os.path.basename(pdf)}))
        doc.close()
    if not docs_list: return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs_list)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(LOCAL_INDEX_FOLDER)
    return vs


vectorstore = load_index()

# כאן אנחנו עוטפים את החיפוש הרגיל ב-MultiQueryRetriever שינסח בעצמו וריאציות לשאלה!
if vectorstore:
    advanced_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )
else:
    advanced_retriever = None


# --- 4. Tools Setup with Self-Correction ---
def search_pdf(query: str):
    if not advanced_retriever:
        return "Error: No professional documents found in the system."

    print(f"\n[RAG DEBUG] Starting Advanced RAG Search for: '{query}'")
    docs = advanced_retriever.invoke(query)

    # מנגנון תיקון עצמי - Self Correction
    if not docs:
        print("[RAG DEBUG] No results found. Prompting agent to try again.")
        return "NO RESULTS FOUND in the knowledge base. SYSTEM COMMAND: You MUST try calling this tool again using different, broader, or alternative agricultural keywords before answering the user."

    return "\n\n".join([d.page_content for d in docs])


def weather_tool_wrapper(city_input: str):
    clean_city = str(city_input).replace("on", "").replace(current_active_date, "").strip()
    full_query = f"{clean_city} on {current_active_date}"
    return weather_service.get_weather(full_query)


tools = [
    Tool(
        name="weather_lookup",
        func=weather_tool_wrapper,
        description="MUST be used for any weather or temperature request."
    ),
    Tool(
        name="agri_knowledge_base",
        func=search_pdf,
        description="Search agricultural manuals. If it returns NO RESULTS, try again with different words."
    )
]

# --- 5. Agent Setup ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """אתה אגרונום מומחה וידידותי בישראל.
    1. בשיחות חולין: ענה בנימוס ואל תשתמש בכלים.
    2. בשאלות מקצועיות: חובה להשתמש בכלי 'weather_lookup'.
    3. ההקשר הנסתר שמועבר אליך מכיל את התאריך ("היום") והמיקום.
    4. חובה להשתמש בכלי 'agri_knowledge_base' לשאלות על גידולים. אם הכלי מחזיר שאין תוצאות, חובה עליך להפעיל אותו שוב עם מילות מפתח אחרות לפחות פעם אחת לפני שאתה מתייאש.
    5. ענה בעברית בלבד ובצורה מקצועית."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent_executor = AgentExecutor(agent=create_openai_tools_agent(llm, tools, prompt), tools=tools, verbose=True)


class ChatRequest(BaseModel):
    user_name: str
    chat_id: str
    query: str
    city: str
    date: str


# --- 6. API Routes ---
@app.get("/")
async def home(): return FileResponse('static/index.html')


@app.post("/get-advice")
async def get_advice(req: ChatRequest):
    global current_active_date
    current_active_date = req.date

    conn = sqlite3.connect(DB_PATH)
    if not conn.execute("SELECT chat_id FROM chats WHERE chat_id = ?", (req.chat_id,)).fetchone():
        title = f"{req.query[:15]}... | {req.city} | {req.date}"
        conn.execute("INSERT INTO chats (chat_id, user_name, title) VALUES (?, ?, ?)",
                     (req.chat_id, req.user_name, title))
        conn.commit()

    hist_rows = conn.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id ASC",
                             (req.chat_id,)).fetchall()
    history = [HumanMessage(content=c) if r == "user" else AIMessage(content=c) for r, c in hist_rows]
    conn.close()

    async def event_generator():
        full_input = f"[הקשר נסתר: התאריך היום הוא {req.date}, המיקום הוא {req.city}]\nהודעת המשתמש: {req.query}"
        final_answer = ""

        try:
            yield f"data: {json.dumps({'type': 'status', 'message': '🤔 מנתח את הבקשה...'})}\n\n"

            async for event in agent_executor.astream_events({"input": full_input, "chat_history": history},
                                                             version="v2"):
                kind = event["event"]

                if kind == "on_tool_start":
                    if event["name"] == "weather_lookup":
                        yield f"data: {json.dumps({'type': 'status', 'message': '🌤️ שולף נתוני אקלים למיקום זה...'})}\n\n"
                    elif event["name"] == "agri_knowledge_base":
                        yield f"data: {json.dumps({'type': 'status', 'message': '📚 מחפש ידע בחקלאות חכמה...'})}\n\n"

                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, 'content') and isinstance(chunk.content, str) and chunk.content:
                        final_answer += chunk.content
                        yield f"data: {json.dumps({'type': 'token', 'text': chunk.content})}\n\n"

            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, 'user', ?)",
                         (req.chat_id, req.query))
            conn.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, 'bot', ?)",
                         (req.chat_id, final_answer))
            conn.commit()
            conn.close()

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'שגיאה בחיבור למודל המחשבה.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/my-chats/{user_name}")
async def get_chats(user_name: str):
    conn = sqlite3.connect(DB_PATH)
    res = conn.execute("SELECT chat_id, title FROM chats WHERE user_name = ?", (user_name,)).fetchall()
    conn.close()
    return [{"chat_id": r[0], "title": r[1]} for r in res]


@app.get("/api/chat-history/{chat_id}")
async def get_hist(chat_id: str):
    conn = sqlite3.connect(DB_PATH)
    res = conn.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,)).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in res]


@app.delete("/api/delete-chat/{chat_id}")
async def del_chat(chat_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)