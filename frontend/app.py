import uuid
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CounselAI | Multi-Agent AI Counsellor",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(circle at 8% 5%, rgba(104, 119, 255, .10), transparent 28%),
                    radial-gradient(circle at 90% 15%, rgba(20, 184, 166, .10), transparent 24%);
    }
    .hero {
        padding: 1.2rem 1.35rem;
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 22px;
        background: rgba(255,255,255,.035);
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2rem; }
    .hero p { margin: 0; opacity: .78; }
    .agent-pill {
        display:inline-block;
        padding:.28rem .58rem;
        margin:.12rem .18rem .12rem 0;
        border-radius:999px;
        border:1px solid rgba(128,128,128,.25);
        font-size:.82rem;
        background:rgba(104,119,255,.08);
    }
    .trace-card {
        padding: .75rem .9rem;
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 14px;
        margin: .45rem 0;
        background: rgba(255,255,255,.025);
    }
    .trace-title { font-weight: 700; margin-bottom: .2rem; }
    .trace-meta { opacity:.72; font-size:.84rem; }
    div[data-testid="stChatMessage"] { border-radius: 18px; }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "candidate_id" not in st.session_state:
    st.session_state.candidate_id = f"candidate-{uuid.uuid4().hex[:8]}"
if "rag_uploads" not in st.session_state:
    st.session_state.rag_uploads = []


def agent_label(name: str) -> str:
    labels = {
        "academic_agent": "📘 Knowledge & Learning Specialist",
        "study_coach": "🗓️ Learning & Productivity Coach",
        "exam_mentor": "📝 Exam & Assessment Mentor",
        "career_advisor": "🧭 Career & Employment Advisor",
        "college_advisor": "🏫 Education & College Advisor",
        "admission_agent": "📄 Admission Specialist",
        "wellbeing_agent": "💚 Wellbeing & Emotional Support Specialist",
        "Supervisor": "🧠 Supervisor",
        "Memory": "🗃️ Candidate Memory",
        "User": "👤 User",
        "Head Counsellor": "💬 Head Counsellor",
    }
    return labels.get(name, name.replace("_", " ").title())


def render_trace(logs):
    if not logs:
        st.info("Internal agent communication will appear after you send a message.")
        return

    for item in logs:
        sender = item.get("sender", "Agent")
        recipient = item.get("recipient")
        item_type = item.get("type", "agent")
        status = item.get("status", "")
        direction = f"{agent_label(sender)} → {agent_label(recipient)}" if recipient else agent_label(sender)

        st.markdown(
            f"<div class='trace-card'><div class='trace-title'>{direction}</div>"
            f"<div class='trace-meta'>{item_type.replace('_', ' ').title()} · {status.replace('_', ' ').title()}</div></div>",
            unsafe_allow_html=True,
        )

        if item.get("query"):
            st.caption(f"Understood query: {item['query']}")
        if item.get("rationale"):
            st.write(item["rationale"])
        if item.get("selected_agents"):
            pills = "".join(
                f"<span class='agent-pill'>{agent_label(a)}</span>" for a in item["selected_agents"]
            )
            st.markdown(pills, unsafe_allow_html=True)

        if item.get("established_facts"):
            st.caption("✅ Established: " + " · ".join(item["established_facts"]))
        if item.get("reasonable_inferences"):
            st.caption("💭 Unconfirmed inference: " + " · ".join(item["reasonable_inferences"]))
        if item.get("open_questions"):
            st.caption("❓ Team should resolve: " + " · ".join(item["open_questions"]))
        if item.get("answered_questions"):
            st.caption("✅ Resolved by this agent: " + " · ".join(item["answered_questions"]))
        if item.get("remaining_questions"):
            st.caption("⏳ Still unresolved: " + " · ".join(item["remaining_questions"]))
        if item.get("recommended_agents"):
            st.caption("↪ Suggested handoff: " + ", ".join(agent_label(a) for a in item["recommended_agents"]))
        if item.get("remaining_question"):
            st.caption("🔎 Highest-value unresolved point: " + item["remaining_question"])
        if item.get("clarification_question"):
            st.warning(item["clarification_question"])

        message = item.get("message") or item.get("content")
        if message and item_type == "agent_message":
            st.markdown(f"> {message}")
        elif message and item_type == "counsellor_message" and recipient != "User":
            st.caption(message)

        if item.get("used_rag"):
            st.caption("📚 Relevant uploaded RAG context was available to this agent")
        if item.get("memory_reused"):
            st.caption("🧠 Relevant candidate-specific remote memory was available")


with st.sidebar:
    st.markdown("## 💬 CounselAI")
    st.caption("Advanced multi-agent counselling workspace")
    st.divider()

    st.markdown("### Candidate session")
    st.session_state.candidate_id = st.text_input(
        "Candidate ID",
        value=st.session_state.candidate_id,
        help="Remote memory and uploaded RAG documents are isolated using this candidate ID.",
    )

    st.markdown("### 📚 RAG knowledge upload")
    uploaded_file = st.file_uploader(
        "Upload candidate/context document",
        type=["pdf"],
        help="Upload counselling notes, learning/career information, reference material or other relevant PDFs.",
    )
    doc_type = st.selectbox(
        "Document type",
        ["candidate_document", "counselling_notes", "learning_notes", "career_profile", "reference_material", "other"],
    )
    if uploaded_file is not None and st.button("Index document", use_container_width=True, type="primary"):
        with st.spinner("Reading and indexing document..."):
            try:
                res = requests.post(
                    f"{API_BASE}/api/rag/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={"candidate_id": st.session_state.candidate_id, "doc_type": doc_type},
                    timeout=120,
                )
                if res.ok:
                    data = res.json()
                    st.session_state.rag_uploads.append(data)
                    st.success(f"Indexed {data.get('chunks_indexed', 0)} chunks from {uploaded_file.name}")
                else:
                    st.error(res.text)
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    if st.session_state.rag_uploads:
        st.caption("Indexed this session")
        for item in st.session_state.rag_uploads[-5:]:
            st.markdown(f"• {item.get('filename')} · {item.get('chunks_indexed', 0)} chunks")

    st.divider()
    if st.button("🗑️ New counselling session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_trace = []
        st.rerun()

st.markdown(
    """
<div class="hero">
    <h1>Advanced AI Counsellor</h1>
    <p>A Head Counsellor works with a Supervisor, candidate-specific memory and a specialist agent panel. Specialists discuss relevant questions internally, resolve what they can from context, memory and RAG, and the Head Counsellor responds naturally or asks one focused clarification when needed.</p>
</div>
""",
    unsafe_allow_html=True,
)

chat_col, trace_col = st.columns([1.65, 1], gap="large")

with chat_col:
    st.markdown("### 💬 Counselling session")
    if not st.session_state.messages:
        st.info("Try: “I lost my job recently and I’m struggling to decide what I should focus on next.”")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("selected_specialists"):
                pills = "".join(
                    f"<span class='agent-pill'>{agent_label(a)}</span>" for a in msg["selected_specialists"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            if msg.get("memory_reused"):
                st.caption("🧠 Previous candidate context used")

with trace_col:
    st.markdown("### 🧠 Agent group chat")
    st.caption("Developer trace of facts, unresolved questions, agent handoffs and sufficiency checks. Private chain-of-thought is not exposed.")
    render_trace(st.session_state.last_trace)

user_input = st.chat_input("What would you like help with?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    payload_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in {"user", "assistant"}
    ]

    try:
        with st.spinner("Counsellor is understanding your message and consulting the right specialists..."):
            response = requests.post(
                f"{API_BASE}/api/chat",
                json={
                    "messages": payload_messages,
                    "candidate_id": st.session_state.candidate_id,
                },
                timeout=180,
            )

        if response.ok:
            data = response.json()
            st.session_state.last_trace = data.get("group_chat_log", [])
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["response"],
                "selected_specialists": data.get("selected_specialists", []),
                "is_incomplete": data.get("is_incomplete", False),
                "memory_reused": data.get("memory_reused", False),
            })
            st.rerun()
        else:
            st.error(f"Backend error ({response.status_code}): {response.text}")
    except Exception as exc:
        st.error(f"Could not connect to the counselling backend: {exc}")
