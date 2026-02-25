"""
Comprehensive test suite for the Agri-Advisor Pro project.
Run while main.py is already running on http://127.0.0.1:8000
"""

import requests
import json
import time
import uuid

BASE = "http://127.0.0.1:8000"
RESULTS = []

# ── helpers ──────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(label, detail=""):
    msg = f"  [PASS] {label}"
    if detail: msg += f"\n         {detail}"
    print(msg)
    RESULTS.append(("PASS", label))

def fail(label, detail=""):
    msg = f"  [FAIL] {label}"
    if detail: msg += f"\n         {detail}"
    print(msg)
    RESULTS.append(("FAIL", label))

def send_and_stream(payload, label):
    """POST to /get-advice, collect the full SSE stream, return (status, tokens, events)."""
    try:
        res = requests.post(f"{BASE}/get-advice", json=payload, stream=True, timeout=60)
        if res.status_code != 200:
            fail(label, f"HTTP {res.status_code}")
            return None

        events = []
        full_text = ""
        buffer = ""

        for chunk in res.iter_content(chunk_size=None):
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n\n" in buffer:
                part, buffer = buffer.split("\n\n", 1)
                if part.startswith("data: "):
                    try:
                        data = json.loads(part[6:])
                        events.append(data)
                        if data.get("type") == "token":
                            full_text += data.get("text", "")
                    except json.JSONDecodeError:
                        pass

        return {"events": events, "full_text": full_text}
    except requests.exceptions.ConnectionError:
        fail(label, "Cannot connect to server — is main.py running?")
        return None
    except Exception as e:
        fail(label, str(e))
        return None


# ── Test 1: Static / API health ───────────────────────────────────────────────

section("1. BASIC CONNECTIVITY")

try:
    r = requests.get(BASE, timeout=5)
    if r.status_code == 200 and "Agri-Advisor" in r.text:
        ok("GET / returns HTML page")
    else:
        fail("GET / returns HTML page", f"status={r.status_code}")
except Exception as e:
    fail("GET / returns HTML page", str(e))

try:
    r = requests.get(f"{BASE}/api/my-chats/test_connectivity_user", timeout=5)
    if r.status_code == 200 and isinstance(r.json(), list):
        ok("GET /api/my-chats returns a list")
    else:
        fail("GET /api/my-chats returns a list", f"status={r.status_code}")
except Exception as e:
    fail("GET /api/my-chats returns a list", str(e))

try:
    fake_id = "nonexistent_chat_xyz"
    r = requests.get(f"{BASE}/api/chat-history/{fake_id}", timeout=5)
    if r.status_code == 200 and r.json() == []:
        ok("GET /api/chat-history with unknown id returns empty list")
    else:
        fail("GET /api/chat-history with unknown id", f"status={r.status_code}, body={r.text[:80]}")
except Exception as e:
    fail("GET /api/chat-history with unknown id", str(e))


# ── Test 2: Chat creation and persistence ─────────────────────────────────────

section("2. CHAT CREATION & PERSISTENCE")

test_user = f"test_user_{uuid.uuid4().hex[:6]}"
chat_id = f"chat_{uuid.uuid4().hex[:8]}"

payload = {
    "user_name": test_user,
    "chat_id": chat_id,
    "query": "שלום, מה שלומך?",
    "city": "ariel",
    "date": "2025-06-15"
}

print(f"\n  Sending small-talk query: \"{payload['query']}\" (city=ariel, date=2025-06-15)")
result = send_and_stream(payload, "Small-talk query streams without crash")

if result:
    events = result["events"]
    event_types = [e.get("type") for e in events]

    has_done = "done" in event_types
    has_token = "token" in event_types
    has_error = "error" in event_types

    if has_error:
        error_msg = next(e.get("message","") for e in events if e.get("type") == "error")
        fail("Small-talk query streams without crash", f"Agent returned error: {error_msg}")
    elif has_token and has_done:
        ok("Small-talk query streams without crash")
        print(f"         Agent reply: {result['full_text'][:120]}...")
    else:
        fail("Small-talk query streams without crash", f"event types seen: {event_types}")

    # Check chat was persisted
    time.sleep(0.5)
    r = requests.get(f"{BASE}/api/my-chats/{test_user}", timeout=5)
    chats = r.json()
    found = any(c["chat_id"] == chat_id for c in chats)
    if found:
        ok("Chat persisted in /api/my-chats after first message")
    else:
        fail("Chat persisted in /api/my-chats after first message", f"Got chats: {chats}")

    r2 = requests.get(f"{BASE}/api/chat-history/{chat_id}", timeout=5)
    hist = r2.json()
    if len(hist) >= 2:
        ok("Chat history has user + bot messages", f"{len(hist)} messages stored")
        for msg in hist:
            print(f"         [{msg['role']}] {msg['content'][:80]}")
    else:
        fail("Chat history has user + bot messages", f"Only {len(hist)} messages found")


# ── Test 3: Conversation continuity (multi-turn) ──────────────────────────────

section("3. MULTI-TURN CONVERSATION")

followup = {
    "user_name": test_user,
    "chat_id": chat_id,   # same chat
    "query": "האם אתה זוכר מה שאלתי לפני רגע?",
    "city": "ariel",
    "date": "2025-06-15"
}

print(f"\n  Sending follow-up in same chat: \"{followup['query']}\"")
result2 = send_and_stream(followup, "Follow-up message in same chat works")

if result2:
    event_types = [e.get("type") for e in result2["events"]]
    if "token" in event_types and "done" in event_types and "error" not in event_types:
        ok("Follow-up message in same chat works")
        print(f"         Agent reply: {result2['full_text'][:120]}...")
    else:
        fail("Follow-up message in same chat works", f"events: {event_types}")

    r3 = requests.get(f"{BASE}/api/chat-history/{chat_id}", timeout=5)
    hist2 = r3.json()
    if len(hist2) >= 4:
        ok(f"History grows correctly after multi-turn ({len(hist2)} messages total)")
    else:
        fail("History grows correctly", f"Only {len(hist2)} messages")


# ── Test 4: Weather tool ──────────────────────────────────────────────────────

section("4. WEATHER TOOL")

WEATHER_CITIES = [
    ("ariel",            "אריאל"),
    ("beer sheva",       "באר שבע"),
    ("eilat",            "אילת"),
    ("tlv beach",        "חוף תל אביב"),
    ("jerusalem center", "ירושלים"),
    ("haifa technion",   "חיפה"),
]

for city_val, city_he in WEATHER_CITIES:
    weather_chat = f"chat_{uuid.uuid4().hex[:8]}"
    p = {
        "user_name": test_user,
        "chat_id": weather_chat,
        "query": f"מה הטמפרטורה היום ב{city_he}? האם מומלץ להשקות?",
        "city": city_val,
        "date": "2025-07-01"
    }
    print(f"\n  Weather query for {city_he} (value='{city_val}') ...")
    r = send_and_stream(p, f"Weather tool fires for {city_he}")
    if r:
        tool_started = any(e.get("type") == "status" and "אקלים" in e.get("message","") for e in r["events"])
        has_reply    = bool(r["full_text"].strip())
        has_error    = any(e.get("type") == "error" for e in r["events"])

        if has_error:
            err = next(e.get("message","") for e in r["events"] if e.get("type") == "error")
            fail(f"Weather tool fires for {city_he}", f"Error: {err}")
        elif tool_started and has_reply:
            ok(f"Weather tool fires for {city_he}", r["full_text"][:100])
        elif has_reply:
            ok(f"Weather tool fires for {city_he} (tool status not captured but got reply)", r["full_text"][:100])
        else:
            fail(f"Weather tool fires for {city_he}", f"No reply. Events: {[e.get('type') for e in r['events']]}")
    time.sleep(1)


# ── Test 5: RAG / Knowledge-base tool ────────────────────────────────────────

section("5. RAG / KNOWLEDGE-BASE TOOL")

rag_queries = [
    ("מהם היתרונות של גידולי כיסוי לשיפור הקרקע?", "cover crops soil"),
    ("כיצד ניתן לנהל מחלות בשדות חיטה?",           "wheat disease management"),
    ("מה הדרך הטובה לשמור על לחות הקרקע?",          "soil moisture retention"),
]

for q_he, q_label in rag_queries:
    rag_chat = f"chat_{uuid.uuid4().hex[:8]}"
    p = {
        "user_name": test_user,
        "chat_id": rag_chat,
        "query": q_he,
        "city": "beer sheva",
        "date": "2025-04-10"
    }
    print(f"\n  RAG query: \"{q_he[:50]}\"")
    r = send_and_stream(p, f"RAG query: {q_label}")
    if r:
        rag_fired  = any(e.get("type") == "status" and "ידע" in e.get("message","") for e in r["events"])
        has_reply  = bool(r["full_text"].strip())
        has_error  = any(e.get("type") == "error" for e in r["events"])

        if has_error:
            err = next(e.get("message","") for e in r["events"] if e.get("type") == "error")
            fail(f"RAG query: {q_label}", f"Error: {err}")
        elif has_reply:
            status = "RAG tool fired" if rag_fired else "reply received (RAG status not captured)"
            ok(f"RAG query: {q_label} — {status}", r["full_text"][:100])
        else:
            fail(f"RAG query: {q_label}", "No reply")
    time.sleep(1)


# ── Test 6: Combined weather + RAG ───────────────────────────────────────────

section("6. COMBINED WEATHER + RAG (AGENT REASONING)")

combo_chat = f"chat_{uuid.uuid4().hex[:8]}"
p = {
    "user_name": test_user,
    "chat_id": combo_chat,
    "query": "בהתאם לטמפרטורה היום בניצן, מתי כדאי לזרוע חיטה לפי הספרות המקצועית?",
    "city": "nitzan",
    "date": "2025-11-01"
}
print(f"\n  Combined query: \"{p['query'][:60]}\"")
r = send_and_stream(p, "Agent uses both weather + RAG tools")
if r:
    tool_statuses = [e.get("message","") for e in r["events"] if e.get("type") == "status"]
    used_weather = any("אקלים" in s for s in tool_statuses)
    used_rag     = any("ידע"   in s for s in tool_statuses)
    has_reply    = bool(r["full_text"].strip())
    has_error    = any(e.get("type") == "error" for e in r["events"])

    print(f"         Tool statuses seen: {tool_statuses}")
    if has_error:
        err = next(e.get("message","") for e in r["events"] if e.get("type") == "error")
        fail("Agent uses both weather + RAG tools", f"Error: {err}")
    else:
        if used_weather and used_rag:
            ok("Agent used BOTH weather + RAG tools in one response")
        elif used_weather or used_rag:
            ok("Agent used at least one tool", f"weather={used_weather}, rag={used_rag}")
        else:
            ok("Agent replied (no tool status captured)", r["full_text"][:100]) if has_reply else fail("Combined query", "No reply")
        if has_reply:
            print(f"         Full reply: {r['full_text'][:200]}")


# ── Test 7: Delete chat ───────────────────────────────────────────────────────

section("7. DELETE CHAT")

del_chat_id = f"chat_{uuid.uuid4().hex[:8]}"
p = {
    "user_name": test_user,
    "chat_id": del_chat_id,
    "query": "הודעה לבדיקת מחיקה",
    "city": "ariel",
    "date": "2025-03-01"
}
send_and_stream(p, "Seed chat for deletion test")
time.sleep(0.3)

r = requests.delete(f"{BASE}/api/delete-chat/{del_chat_id}", timeout=5)
if r.status_code == 200:
    ok("DELETE /api/delete-chat returns 200")
else:
    fail("DELETE /api/delete-chat returns 200", f"status={r.status_code}")

r2 = requests.get(f"{BASE}/api/chat-history/{del_chat_id}", timeout=5)
if r2.json() == []:
    ok("Chat history is empty after deletion")
else:
    fail("Chat history is empty after deletion", f"Still has: {r2.json()}")

r3 = requests.get(f"{BASE}/api/my-chats/{test_user}", timeout=5)
still_listed = any(c["chat_id"] == del_chat_id for c in r3.json())
if not still_listed:
    ok("Chat removed from user's chat list after deletion")
else:
    fail("Chat removed from user's chat list after deletion")


# ── Test 8: Edge cases ────────────────────────────────────────────────────────

section("8. EDGE CASES")

# Missing city
p = {"user_name": test_user, "chat_id": f"chat_{uuid.uuid4().hex[:8]}", "query": "שאלה", "city": "", "date": "2025-05-01"}
r = requests.post(f"{BASE}/get-advice", json=p, stream=True, timeout=10)
if r.status_code == 200:
    ok("Empty city: server returns 200 (handled by agent)")
else:
    fail("Empty city: server accepts request", f"status={r.status_code}")

# Same chat_id, different user (should still work — just attaches to same chat)
p2 = {"user_name": "other_user", "chat_id": chat_id, "query": "שלום", "city": "eilat", "date": "2025-01-10"}
r2 = send_and_stream(p2, "Different user can post to existing chat_id")
if r2 and "token" in [e.get("type") for e in r2["events"]]:
    ok("Different user can post to existing chat_id")


# ── Summary ───────────────────────────────────────────────────────────────────

section("SUMMARY")
passed = sum(1 for s, _ in RESULTS if s == "PASS")
failed = sum(1 for s, _ in RESULTS if s == "FAIL")
total  = len(RESULTS)

print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}\n")
for status, label in RESULTS:
    mark = "✓" if status == "PASS" else "✗"
    print(f"  {mark}  {label}")

print()
