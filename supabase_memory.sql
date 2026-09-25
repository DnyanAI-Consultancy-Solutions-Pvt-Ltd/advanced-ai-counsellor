-- Run once in your Supabase SQL editor.
-- Candidate-specific counselling memory. No local persistence is used by memory_tool.py.
create extension if not exists pgcrypto;

create table if not exists public.counselling_memories (
    id uuid primary key default gen_random_uuid(),
    candidate_id text not null,
    user_message text not null,
    counsellor_response text not null,
    query_summary text default '',
    established_facts jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists counselling_memories_candidate_created_idx
    on public.counselling_memories(candidate_id, created_at desc);

alter table public.counselling_memories enable row level security;

-- Backend should use SUPABASE_SERVICE_ROLE_KEY server-side only.
-- Do not expose the service-role key in Streamlit/browser code.
