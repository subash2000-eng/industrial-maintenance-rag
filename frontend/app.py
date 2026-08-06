import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Industrial Maintenance RAG",
    page_icon ="🔧",
    layout    ="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 100%);
        color: #e2e8f0;
    }
    .hero {
        background   : linear-gradient(135deg, #1e3a5f, #0f2744);
        border-left  : 5px solid #f97316;
        border-radius: 12px;
        padding      : 2rem;
        margin-bottom: 2rem;
    }
    .hero h1 { color: #f97316; font-size: 1.8rem; margin: 0 0 0.5rem; }
    .hero p  { color: #94a3b8; margin: 0; }
    .answer-card {
        background   : #0f2034;
        border-left  : 5px solid #22c55e;
        border-radius: 10px;
        padding      : 1.5rem;
        margin       : 1rem 0;
        line-height  : 1.8;
    }
    .source-card {
        background   : #1e2d3d;
        border       : 1px solid #2d4a7a;
        border-radius: 8px;
        padding      : 1rem;
        margin       : 0.5rem 0;
    }
    .badge {
        display      : inline-block;
        padding      : 2px 10px;
        border-radius: 20px;
        font-size    : 0.75rem;
        font-weight  : 600;
    }
    .badge-high   { background: #166534; color: #4ade80; }
    .badge-medium { background: #713f12; color: #fbbf24; }
    .badge-low    { background: #7f1d1d; color: #f87171; }
    .history-item {
        background   : #1e2d3d;
        border-radius: 6px;
        padding      : 0.7rem;
        margin       : 0.3rem 0;
        font-size    : 0.82rem;
    }
    .stButton > button {
        background   : linear-gradient(135deg, #f97316, #ea580c);
        color        : white;
        border       : none;
        border-radius: 8px;
        font-weight  : 600;
        width        : 100%;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628, #0f1e35);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .stTextArea textarea {
        background   : #1e2d3d !important;
        color        : #e2e8f0 !important;
        border       : 1px solid #2d4a7a !important;
        border-radius: 8px !important;
    }
    .stSelectbox > div > div {
        background: #1e2d3d !important;
        color     : #e2e8f0 !important;
    }
    div[data-testid="stMetric"] {
        background   : #1e2d3d;
        border       : 1px solid #2d4a7a;
        border-radius: 8px;
        padding      : 0.8rem;
    }
    div[data-testid="stMetricValue"] { color: #f97316 !important; }
    .section-title {
        color       : #f97316;
        font-size   : 1rem;
        font-weight : 600;
        padding-left: 0.5rem;
        border-left : 3px solid #f97316;
        margin      : 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def check_health() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def get_manuals_fresh() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/manuals", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def query_rag(query, top_k_retrieve=20, top_k_rerank=3, manual_filter=None) -> dict:
    payload = {
        "query"         : query,
        "top_k_retrieve": top_k_retrieve,
        "top_k_rerank"  : top_k_rerank
    }
    if manual_filter and manual_filter != "All Manuals":
        payload["manual_filter"] = manual_filter
    try:
        r = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=300)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def upload_manual(file_bytes, filename) -> dict:
    try:
        r = requests.post(
            f"{API_BASE_URL}/upload-manual",
            files  ={"file": (filename, file_bytes, "application/pdf")},
            timeout=600
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def submit_feedback(query_id, rating, was_helpful, comment=None):
    try:
        requests.post(
            f"{API_BASE_URL}/feedback",
            json   ={"query_id": query_id, "rating": rating,
                     "was_helpful": was_helpful, "comment": comment},
            timeout=10
        )
    except Exception:
        pass


def get_history() -> list:
    try:
        r = requests.get(f"{API_BASE_URL}/history?limit=10", timeout=10)
        return r.json().get("recent_queries", []) if r.status_code == 200 else []
    except Exception:
        return []


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:2rem;">🔧</div>
            <div style="color:#f97316; font-weight:700;">Maintenance RAG</div>
            <div style="color:#64748b; font-size:0.8rem;">AI Repair Assistant</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)

        health = check_health()
        if health:
            status = health.get("status", "unknown")
            groq   = health.get("ollama_running", False)
            chunks = health.get("chroma_chunks", 0)
            model  = health.get("model", "N/A")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"{'🟢' if status == 'ready' else '🟡'} **API**<br>"
                    f"<small>{status.title()}</small>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"{'🟢' if groq else '🔴'} **Groq**<br>"
                    f"<small>{'On' if groq else 'Off'}</small>",
                    unsafe_allow_html=True
                )
            st.markdown(f"""
            <div style="background:#0f2034; border:1px solid #2d4a7a;
                        border-radius:6px; padding:0.7rem; margin-top:0.5rem;">
                <div style="color:#64748b; font-size:0.75rem;">CHUNKS INDEXED</div>
                <div style="color:#f97316; font-size:1.3rem; font-weight:700;">{chunks:,}</div>
                <div style="color:#64748b; font-size:0.75rem;">Model: {model}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("API not reachable.\n\nRun: python api/main.py")

        st.markdown("---")
        st.markdown('<div class="section-title">Upload Manual</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose PDF", type=['pdf'], label_visibility="collapsed", key="pdf_uploader")

        if uploaded_file is not None:
            st.markdown(f"""
            <div style="background:#0f2034; border:1px solid #2d4a7a;
                        border-radius:6px; padding:0.7rem; margin-bottom:0.5rem;">
                📄 <strong>{uploaded_file.name}</strong><br>
                <small style="color:#64748b;">{uploaded_file.size/1024/1024:.1f} MB</small>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Ingest Manual", use_container_width=True, key="ingest_btn"):
                with st.spinner(f"Ingesting {uploaded_file.name}... Please wait."):
                    result = upload_manual(uploaded_file.getvalue(), uploaded_file.name)

                if "error" in result:
                    st.error(f"Failed: {result['error']}")
                else:
                    new_manual = uploaded_file.name.replace(".pdf", "")
                    all_names  = result.get("manual_names", [])
                    updated    = [new_manual] + [m for m in all_names if m != new_manual]
                    st.session_state["manual_list"] = updated
                    st.success(
                        f"Ingested successfully.\n\n"
                        f"Pages: {result['pages']} | "
                        f"Chunks: {result['chunks_added']} | "
                        f"Total: {result['total_chunks']}"
                    )
                    st.rerun()

        st.markdown("---")
        st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
        top_k_retrieve = st.slider("Chunks to retrieve", 5, 50, 20)
        top_k_rerank   = st.slider("Chunks after ranking", 1, 10, 3)

        st.markdown("---")
        st.markdown('<div class="section-title">Filter Manual</div>', unsafe_allow_html=True)

        manuals_data = get_manuals_fresh()
        api_manuals  = [m["manual_name"] for m in manuals_data.get("manuals", [])]
        session_list = st.session_state.get("manual_list", [])
        combined     = list(dict.fromkeys(session_list + api_manuals))
        manual_names = ["All Manuals"] + combined

        manual_filter = st.selectbox("Select manual", manual_names, label_visibility="collapsed")

        if manual_filter != "All Manuals":
            st.markdown(
                f'<div style="background:#0f2034; border:1px solid #f97316; '
                f'border-radius:6px; padding:0.5rem; font-size:0.8rem; color:#f97316;">'
                f'Searching only: <strong>{manual_filter}</strong></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown('<div class="section-title">Recent Queries</div>', unsafe_allow_html=True)

        history = get_history()
        if history:
            for item in history[:5]:
                st.markdown(
                    f'<div class="history-item">🔍 {item.get("query", "")[:40]}...<br>'
                    f'<span style="color:#64748b; font-size:0.75rem;">'
                    f'{item.get("confidence", "")} | {item.get("response_time_s", 0):.0f}s'
                    f'</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("No queries yet.")

    return top_k_retrieve, top_k_rerank, manual_filter


def render_result(result: dict):
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return

    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Time", f"{result.get('response_time_s', 0):.1f}s")
    with col2:
        st.metric("Chunks", result.get("chunks_used", 0))
    with col3:
        st.metric("Confidence", result.get("confidence", "N/A"))
    with col4:
        st.metric("Model", result.get("model", "N/A").split("/")[-1])
    with col5:
        st.metric("Language", result.get("detected_language", "English"))

    lang = result.get("detected_language", "English")
    if lang != "English":
        st.info(f"Query detected in {lang}. Searched in English. Answer translated to {lang}.")

    st.markdown("---")
    st.markdown('<div class="section-title">Repair Instructions</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-card">{result.get("answer", "No answer generated.")}</div>',
        unsafe_allow_html=True
    )

    sources = result.get("sources", [])
    if sources:
        st.markdown("---")
        st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)
        for s in sources:
            conf  = s.get("confidence", "Unknown")
            badge = "badge-high" if conf == "High" else "badge-medium" if conf == "Medium" else "badge-low"
            st.markdown(f"""
            <div class="source-card">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                    <strong style="color:#f97316;">[SOURCE {s['source_num']}]</strong>
                    <span class="badge {badge}">{conf}</span>
                </div>
                <div>📖 <strong>{s['manual_name']}</strong></div>
                <div style="color:#94a3b8;">Page {s['page_number']} · {s['section_title']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Rate This Answer</div>', unsafe_allow_html=True)
    query_id = result.get("query_id", "unknown")
    col1, col2 = st.columns(2)
    with col1:
        rating = st.slider("Rating", 1, 5, 4, key=f"r_{query_id}")
    with col2:
        helpful = st.radio("Helpful?", ["Yes", "No"], key=f"h_{query_id}")
        comment = st.text_input("Comment", key=f"c_{query_id}")

    if st.button("Submit Feedback", key=f"fb_{query_id}"):
        submit_feedback(query_id, rating, helpful == "Yes", comment or None)
        st.success("Feedback submitted.")


def render_welcome():
    manuals_data  = get_manuals_fresh()
    total_chunks  = manuals_data.get("total_chunks", 0)
    total_manuals = manuals_data.get("total_manuals", 0)
    manuals_list  = manuals_data.get("manuals", [])

    if total_chunks > 0:
        st.markdown(f"""
        <div style="background:#0f2034; border:1px solid #2d4a7a;
                    border-radius:10px; padding:1.2rem; margin-bottom:1.2rem;">
            <div style="color:#22c55e; font-weight:600; margin-bottom:0.8rem;">System Ready</div>
            <div style="display:flex; gap:2rem;">
                <div>
                    <div style="color:#f97316; font-size:1.5rem; font-weight:700;">{total_manuals}</div>
                    <div style="color:#64748b; font-size:0.8rem;">Manuals</div>
                </div>
                <div>
                    <div style="color:#f97316; font-size:1.5rem; font-weight:700;">{total_chunks:,}</div>
                    <div style="color:#64748b; font-size:0.8rem;">Chunks Indexed</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for m in manuals_list:
            st.markdown(f"""
            <div style="background:#1e2d3d; border:1px solid #2d4a7a;
                        border-radius:6px; padding:0.7rem; margin:0.3rem 0;
                        display:flex; justify-content:space-between;">
                <span>📖 {m['manual_name']}</span>
                <span style="color:#f97316;">{m['chunk_count']} chunks</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1e2034; border:2px dashed #2d4a7a;
                    border-radius:10px; padding:2rem; text-align:center;">
            <div style="font-size:2.5rem; margin-bottom:0.8rem;">📤</div>
            <div style="color:#f97316; font-weight:600; margin-bottom:0.4rem;">No Manuals Yet</div>
            <div style="color:#64748b;">Upload a PDF manual using the sidebar to get started.</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    if "manual_list" not in st.session_state:
        st.session_state["manual_list"] = []

    top_k_retrieve, top_k_rerank, manual_filter = render_sidebar()

    st.markdown("""
    <div class="hero">
        <h1>🔧 Industrial Maintenance RAG</h1>
        <p>Describe any machine fault in plain English or your language —
        get instant cited repair instructions from your maintenance manuals.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Describe the Fault</div>', unsafe_allow_html=True)

    examples = [
        "Select an example...",
        "Motor bearing overheating after 2 hours of operation",
        "Compressor not reaching target pressure",
        "Excessive vibration and noise from motor",
        "Oil leaking from compressor seal area",
        "Motor fails to start under load"
    ]
    selected = st.selectbox("Examples", examples, label_visibility="collapsed")
    default  = "" if selected == "Select an example..." else selected

    query = st.text_area(
        "Fault",
        value      =default,
        height     =130,
        placeholder=(
            "Describe the fault in detail...\n"
            "You can also type in Tamil, Hindi, or any language."
        ),
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        submit = st.button("Get Repair Instructions", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    if submit:
        if not query or len(query.strip()) < 5:
            st.warning("Please enter at least 5 characters.")
            return

        with st.spinner("Searching manuals and generating answer..."):
            result = query_rag(
                query         =query.strip(),
                top_k_retrieve=top_k_retrieve,
                top_k_rerank  =top_k_rerank,
                manual_filter =manual_filter
            )
        render_result(result)
    else:
        render_welcome()


if __name__ == "__main__":
    main()