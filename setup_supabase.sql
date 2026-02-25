-- Run this in Supabase → SQL Editor to create the required tables.
-- After creating, go to Authentication → Policies and disable RLS on both tables,
-- OR use the service role key (recommended for server-side use).

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
