# Candidate-specific remote memory setup

This version keeps counselling memory in Supabase, not in local files.

1. Create a Supabase project.
2. Open the SQL editor and run `supabase_memory.sql` once.
3. Copy `.env.example` to `.env`.
4. Set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in the backend environment.
5. Keep the service-role key server-side only. Never put it in Streamlit/browser JavaScript.
6. Use a stable `candidate_id` for the same person across sessions. Different candidate IDs are isolated by lookup.

## Memory behavior

- Before the Supervisor runs, the Memory node retrieves only that candidate's recent remote counselling history.
- A strongly similar repeated concern is flagged to the Supervisor. The Supervisor may route directly to the Head Counsellor only when there is no meaningful new information.
- If circumstances changed, normal specialist group-chat remains active and only relevant specialists are consulted.
- After the Head Counsellor responds, the completed turn is saved remotely.
- If Supabase is not configured or temporarily unavailable, the counselling flow continues without persistent memory.

The current implementation uses lightweight text similarity over candidate-scoped recent remote records, so it does not require a local embedding model or local vector database. It can later be upgraded to pgvector embeddings without changing the graph contract.
