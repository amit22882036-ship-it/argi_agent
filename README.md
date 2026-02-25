# Agri-Advisor Pro

An AI-powered agricultural advisory agent for Israeli farmers, built with LangChain, FastAPI, Pinecone, and Supabase. Deployed on Render.

---

## Project Requirements Coverage

| Requirement | Implementation |
|---|---|
| FastAPI backend | `main.py` — serves all API endpoints |
| LangChain AgentExecutor | `create_openai_tools_agent` with two tools |
| Supabase (PostgreSQL) | Chat and message history persistence |
| Pinecone vector DB | RAG over 13 agricultural PDFs (6,380 chunks) |
| Deployed on Render | Live at `https://agents-960290.onrender.com` |
| `GET /api/team_info` | Group info and student list |
| `GET /api/agent_info` | Agent description, prompt template, worked examples |
| `GET /api/model_architecture` | Architecture diagram (PNG) |
| `POST /api/execute` | Main agent endpoint with steps logging |
| Module names in steps | `AgentLLM`, `WeatherTool`, `AgriKnowledgeBase` |
| Multi-turn conversation | `chat_id` parameter with Supabase history |
| Hebrew responses | System prompt and all agent output in Hebrew |

---

## Features Beyond Base Requirements

- **Rich weather context** — the weather tool returns not just today's conditions but rolling 7-day and 30-day summaries (avg/max/min temperature, total rainfall, humidity, frost days), giving the agent historical context for long-term agricultural planning decisions.
- **Smart weather caching** — on first query for a city, the full hourly JSON is loaded and aggregated down to one row per day (~1,800 rows). All subsequent queries for any date in that city return in under 2ms.
- **Full weather columns** — all 16 measurement columns (TD, TDmax, TDmin, RH, Rain, WS, WSmax, WD, STDwd, etc.) are preserved and passed to the agent.
- **Partial window handling** — if fewer than 7 or 30 days of history exist for a date, the system reports however many days are actually available.
- **Background initialisation** — the server binds its port immediately on startup; Supabase, LLM, embeddings, and Pinecone connect in a background thread so the service is never slow to come up.
- **Lazy Pinecone indexing** — PDFs are only indexed if the Pinecone index is empty; subsequent restarts skip indexing entirely.
- **Multi-city weather coverage** — 16 Israeli weather stations, matched by fuzzy name lookup.

---

## Knowledge Base Data

### Agricultural PDFs (`project_sources/`) — 137 MB, 13 files

| File | Subject |
|---|---|
| `Building-Soils-for-Better-Crops.pdf` | Soil health, organic matter, fertility management |
| `Managing-Cover-Crops-Profitably.pdf` | Cover crop selection, planting, integration |
| `cc3338en.pdf` | FAO good agricultural practices (GAP) guide |
| `einboeck_source_1.pdf` | Mechanical weed control in field crops |
| `source2.pdf` – `source12.pdf` | Additional agronomy sources (irrigation, diseases, soil, water management in Israel) |

All PDFs are chunked (1,000 tokens, 200 overlap) and embedded with `BAAI/bge-small-en-v1.5` (384 dimensions) into Pinecone. **6,380 vectors** total.

### Weather Data (`city_data/`) — 212 MB, 16 files

Hourly meteorological readings from 16 Israeli weather stations, covering multiple years. Each file contains columns: `date`, `TD` (temperature), `TDmax`, `TDmin`, `RH` (humidity), `Rain`, `WS` (wind speed), `WSmax`, `WD` (wind direction), `STDwd`, and others.

| Station | Station |
|---|---|
| Ariel | Ashdod |
| Ashkelon | Avne Eitan |
| Beer Sheva | Eilat |
| Hadera | Haifa Technion |
| Haifa Bate Zakuk | Jerusalem Center |
| Lev Kineret | Maale Gilboa |
| Nitzan | Tel Aviv Beach |
| Yotvata | Zichron Yaakov |

---

## System Pipeline

```
User message (prompt + city + date + optional chat_id)
        │
        ▼
POST /api/execute  (FastAPI)
        │
        ├─ Retrieve chat history from Supabase (if chat_id)
        │
        ▼
LangChain AgentExecutor
        │
        ├─── AgentLLM decides which tools to call
        │
        ├─── WeatherTool (if weather/climate question)
        │         └─ Load city daily cache (<2ms if warm)
        │             Return: today + 7-day + 30-day summaries
        │
        ├─── AgriKnowledgeBase (if agronomic question)
        │         └─ MultiQueryRetriever → 3 sub-queries via LLM
        │             → Pinecone vector search (k=3 each)
        │             → Return top relevant PDF chunks
        │
        └─── AgentLLM synthesises final Hebrew response
                │
                ▼
        Save user + bot messages to Supabase (if chat_id)
                │
                ▼
        Return JSON:
        {
          "status": "ok",
          "error": null,
          "response": "...",   ← Hebrew answer
          "steps": [           ← one entry per tool call / LLM step
            {"module": "WeatherTool",      "prompt": "...", "response": "..."},
            {"module": "AgriKnowledgeBase","prompt": "...", "response": "..."},
            {"module": "AgentLLM",         "prompt": "...", "response": "..."}
          ]
        }
```

---

## Setup on Render

### Prerequisites
- [Render](https://render.com) account
- [Supabase](https://supabase.com) project with the tables below
- [Pinecone](https://pinecone.io) project with a `agri-advisor` index (dimension 384, cosine)
- An OpenAI-compatible LLM API key (project uses [llmod.ai](https://llmod.ai))

### 1. Supabase tables

Run in the Supabase SQL Editor:

```sql
CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    user_name TEXT,
    title TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    chat_id TEXT,
    role TEXT,
    content TEXT
);
ALTER TABLE chats DISABLE ROW LEVEL SECURITY;
ALTER TABLE messages DISABLE ROW LEVEL SECURITY;
```

### 2. Render web service

1. Create a new **Web Service** and connect this repository.
2. Set the following environment variables in the Render dashboard:

| Variable | Value |
|---|---|
| `LLMOD_API_KEY` | Your LLM API key |
| `LLMOD_API_BASE` | `https://api.llmod.ai/v1` (or your endpoint) |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase anon/service key |
| `PINECONE_API_KEY` | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | `agri-advisor` |

3. Render will use `render.yaml` automatically — no further configuration needed.
4. On first deploy, the service will index all PDFs into Pinecone (one-time, ~2–5 minutes). Subsequent deploys skip this step.

### 3. Populate Pinecone locally (recommended)

To avoid the first-deploy indexing load on the free tier, index locally before deploying:

```bash
pip install -r requirements.txt
# set env vars in .env, then:
python - <<'EOF'
# run the indexing block from main.py build_index() with your local .env
EOF
```

Or simply deploy and wait — the agent responds normally while indexing runs in the background.
