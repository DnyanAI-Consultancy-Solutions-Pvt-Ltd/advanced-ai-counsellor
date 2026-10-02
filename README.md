---
title: Advanced AI Counsellor
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---


# Advanced AI Student Counsellor

A LangGraph-based multi-agent counselling application with a Head Counsellor, Supervisor, specialist panel, transparent orchestration trace, and per-student RAG document upload.

## How a turn works

1. **Supervisor** reads the conversation, summarizes the student's need, checks whether essential context is missing, and selects zero, one, or multiple specialists.
2. If essential information is missing, the **Head Counsellor** asks one focused clarification question. Specialists are not unnecessarily invoked.
3. When the request is complete, the selected specialist agents contribute domain notes to the internal panel.
4. The **Head Counsellor always participates** and synthesizes the final student-facing guidance.
5. The frontend displays a safe orchestration audit trail showing routing, selected agents, specialist contributions, and whether uploaded RAG context was used.

## Specialist agents

- Academic Specialist
- Study Coach
- Exam Mentor
- Career Advisor
- College Advisor
- Admission Specialist
- Wellbeing Specialist
- Head Counsellor (always user-facing)

## RAG

PDF documents can be uploaded from the Streamlit sidebar. They are chunked, embedded locally with `sentence-transformers/all-MiniLM-L6-v2`, stored in Chroma, and filtered by `student_id` during retrieval.

## Run

Create a `.env` file containing at least:

```env
GROQ_API_KEY=your_key_here
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start backend from the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

Start frontend in a second terminal:

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL shown in the terminal.

## API examples

### Chat

`POST /api/chat`

```json
{
  "student_id": "student-123",
  "messages": [
    {"role": "user", "content": "I am in 12th and confused between CSE and ECE."}
  ]
}
```

### RAG upload

`POST /api/rag/upload` as multipart form data with:

- `file`: PDF
- `student_id`: the student's session ID
- `doc_type`: marksheet / academic_notes / college_info / etc.

## Important design note

The group-chat UI exposes routing decisions and concise agent contributions for debugging and observability. It intentionally does not expose private hidden chain-of-thought.

## Conversational counselling + internal agent chat

The application now separates the student experience from developer observability:

- Student-facing replies are normally 2-5 sentences and ask at most one question per turn.
- The Supervisor checks whether one essential clarification is needed before routing.
- Selected specialists run sequentially and can read concise messages from specialists who ran before them.
- The internal trace records sender, recipient, routing reason, specialist message and RAG usage.
- The Head Counsellor always owns the final response and does not expose internal agent names to the student.
- Internal logs intentionally contain concise conclusions/recommendations, not private model chain-of-thought.

To observe agent communication, run the backend with INFO logging and watch the terminal for lines beginning with `AGENT CHAT`.

## Agentic information-resolution flow

The counselling graph now follows an evidence-first collaboration loop rather than asking the student for missing information immediately:

1. **Supervisor** extracts the actual counselling need, established facts, reasonable-but-unconfirmed inferences, and only high-value unresolved questions.
2. **Specialist panel** receives those open questions. Each specialist can resolve a question, refine another agent's conclusion, or hand a remaining question to another relevant specialist.
3. Specialists can dynamically recommend another agent from the registered specialist pool; the collaboration loop consults that agent before involving the student.
4. **Information Sufficiency Judge** reviews conversation history, specialist conclusions and RAG evidence. It asks the student for clarification only when one unresolved fact is truly blocking useful counselling.
5. **Head Counsellor** acknowledges relevant emotional context and responds conversationally. Clarifications are minimal and may use yes/no or a simple choice when validating an inference.

The developer trace exposes concise conclusions, facts, open questions, handoffs and sufficiency decisions. It intentionally does not expose private chain-of-thought.

## v4: General AI Counsellor + remote candidate memory

This build generalizes student-only wording to an AI Counselling System for people across ages, professions and life situations while preserving the internal Supervisor → specialist group chat → sufficiency check → Head Counsellor flow.

Persistent counselling memory is candidate-specific and remote-only. See `MEMORY_SETUP.md` and `supabase_memory.sql`. The existing RAG subsystem is separate from counselling memory.
