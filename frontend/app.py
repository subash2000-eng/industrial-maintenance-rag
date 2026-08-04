"""
Phase 5 - Streamlit Frontend
Updated with proper manual dropdown refresh after upload.
Newly uploaded manuals appear at TOP of dropdown.
"""

import os
import requests
import streamlit as st

API_BASE_URL = "https://industrial-maintenance-rag.onrender.com"

st.set_page_config(
    page_title="Maintenance RAG",
    page_icon ="🔧",
    layout    ="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 100%);
        color: #e2e8f0;
    }
    .hero {
        background   : linear-gradient(135deg, #1e3a5f 0%, #0f2744 50%, #1a1f3e 100%);
        border       : 1px solid #2d4a7a;
        border-left  : 5px solid #f97316;
        border-radius: 16px;
        padding      : 2.5rem;
        margin-bottom: 2rem;
        position     : relative;
        overflow     : hidden;
    }
    .hero::before {
        content  : "🏭";
        position : absolute;
        right    : 2rem;
        top      : 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity  : 0.15;
    }
    .hero h1 {
        color      : #f97316;
        font-size  : 2rem;
        font-weight: 700;
        margin     : 0 0 0.5rem;
    }
    .hero p {
        color    : #94a3b8;
        font-size: 1rem;
        margin   : 0;
        max-width: 600px;
    }
    .answer-card {
        background   : linear-gradient(135deg, #0f2034, #1a2744);
        border       : 1px solid #2d4a7a;
        border-left  : 5px solid #22c55e;
        border-radius: 12px;
        padding      : 2rem;
        margin       : 1.5rem 0;
        line-height  : 1.8;
    }
    .source-card {
        background   : #1e2d3d;
        border       : 1px solid #2d4a7a;
        border-radius: 10px;
        padding      : 1.2rem;
        margin       : 0.8rem 0;
        transition   : border-color 0.2s;
    }
    .source-card:hover { border-color: #f97316; }
    .badge {
        display      : inline-block;
        padding      : 3px 12px;
        border-radius: 20px;
        font-size    : 0.75rem;
        font-weight  : 600;
    }
    .badge-green  { background:#166534; color:#4ade80; }
    .badge-blue   { background:#1e3a5f; color:#60a5fa; }
    .badge-yellow { background:#713f12; color:#fbbf24; }
    .badge-red    { background:#7f1d1d; color:#f87171; }
    .history-item {
        background   : #1e2d3d;
        border       : 1px solid #2d4a7a;
        border-radius: 8px;
        padding      : 0.8rem;
        margin       : 0.4rem 0;
        font-size    : 0.82rem;
    }
    .stButton > button {
        background : linear-gradient(135deg, #f97316, #ea580c);
        color      : white;
        border     : none;
        border-radius: 10px;
        font-weight: 600;
        font-size  : 1rem;
        width      : 100%;
        box-shadow : 0 4px 15px rgba(249, 115, 22, 0.3);
    }
    .stButton > button:hover {
        transform : translateY(-2px);
        box-shadow: 0 6px 20px rgba(249, 115, 22, 0.4);
    }
    section[data-testid="stSidebar"] {
        background  : linear-gradient(180deg, #0a1628 0%, #0f1e35 100%);
        border-right: 1px solid #1e3a5f;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .stTextArea textarea {
        background   : #1e2d3d !important;
        color        : #e2e8f0 !important;
        border       : 1px solid #2d4a7a !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus {
        border-color: #f97316 !important;
    }
    .stSelectbox > div > div {
        background: #1e2d3d !important;
        border    : 1px solid #2d4a7a !important;
        color     : #e2e8f0 !important;
    }
    div[data-testid="stMetric"] {
        background   : #1e2d3d;
        border       : 1px solid #2d4a7a;
        border-radius: 10px;
        padding      : 1rem;
    }
    div[data-testid="stMetricValue"] { color: #f97316 !important; }
    .section-title {
        color        : #f97316;
        font-size    : 1.1rem;
        font-weight  : 600;
        margin-bottom: 0.8rem;
        padding-left : 0.5rem;
        border-left  : 3px solid #f97316;
    }
    hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# API Helpers
# ════════════════════════════════════════════════════════

def check_health() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def get_manuals_fresh() -> dict:
    """
    Always fetch fresh manual list directly from API.
    Never cached — reads live ChromaDB every time.
    """
    try:
        r = requests.get(
            f"{API_BASE_URL}/manuals",
            timeout=10
        )
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def query_rag(
    query, top_k_retrieve=20,
    top_k_rerank=3, manual_filter=None
) -> dict:
    payload = {
        "query"         : query,
        "top_k_retrieve": top_k_retrieve,
        "top_k_rerank"  : top_k_rerank
    }
    if manual_filter and manual_filter != "All Manuals":
        payload["manual_filter"] = manual_filter
    try:
        r = requests.post(
            f"{API_BASE_URL}/query",
            json   =payload,
            timeout=300
        )
        return r.json() if r.status_code == 200 \
               else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def upload_manual(file_bytes, filename) -> dict:
    try:
        files = {
            "file": (filename, file_bytes, "application/pdf")
        }
        r = requests.post(
            f"{API_BASE_URL}/upload-manual",
            files  =files,
            timeout=600
        )
        return r.json() if r.status_code == 200 \
               else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def submit_feedback(
    query_id, rating, was_helpful, comment=None
):
    try:
        requests.post(
            f"{API_BASE_URL}/feedback",
            json={
                "query_id"   : query_id,
                "rating"     : rating,
                "was_helpful": was_helpful,
                "comment"    : comment
            },
            timeout=10
        )
    except Exception:
        pass


def get_history() -> list:
    try:
        r = requests.get(
            f"{API_BASE_URL}/history?limit=10",
            timeout=10
        )
        return r.json().get("recent_queries", []) \
               if r.status_code == 200 else []
    except Exception:
        return []


# ════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:

        # ── Logo ─────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:2.5rem;">🔧</div>
            <div style="color:#f97316; font-weight:700;
                        font-size:1.1rem;">Maintenance RAG</div>
            <div style="color:#64748b; font-size:0.8rem;">
                AI Repair Assistant
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── System Status ─────────────────────────────────
        st.markdown(
            '<div class="section-title">System Status</div>',
            unsafe_allow_html=True
        )

        health = check_health()

        if health:
            status = health.get("status", "unknown")
            ollama = health.get("ollama_running", False)
            chunks = health.get("chroma_chunks", 0)
            model  = health.get("model", "N/A")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"{'🟢' if status=='ready' else '🟡'} "
                    f"**API**<br>"
                    f"<small>{status.title()}</small>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"{'🟢' if ollama else '🔴'} "
                    f"**Ollama**<br>"
                    f"<small>{'On' if ollama else 'Off'}</small>",
                    unsafe_allow_html=True
                )

            st.markdown(f"""
            <div style="background:#0f2034;
                        border:1px solid #2d4a7a;
                        border-radius:8px;
                        padding:0.8rem;
                        margin-top:0.8rem;">
                <div style="color:#64748b; font-size:0.75rem;">
                    CHUNKS INDEXED
                </div>
                <div style="color:#f97316; font-size:1.5rem;
                            font-weight:700;">{chunks:,}</div>
                <div style="color:#64748b; font-size:0.75rem;
                            margin-top:0.3rem;">
                    Model: {model}
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error(
                "❌ API not reachable\n\n"
                "Run:\n```\npython api/main.py\n```"
            )

        st.markdown("---")

        # ── Upload Manual ─────────────────────────────────
        st.markdown(
            '<div class="section-title">📤 Upload Manual</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose PDF",
            type            =['pdf'],
            label_visibility="collapsed",
            key             ="pdf_uploader"
        )

        if uploaded_file is not None:
            st.markdown(f"""
            <div style="background:#0f2034;
                        border:1px solid #2d4a7a;
                        border-radius:8px;
                        padding:0.8rem;
                        margin-bottom:0.8rem;">
                📄 <strong>{uploaded_file.name}</strong><br>
                <small style="color:#64748b;">
                    {uploaded_file.size/1024/1024:.1f} MB
                </small>
            </div>
            """, unsafe_allow_html=True)

            if st.button(
                "⬆️ Ingest Manual",
                use_container_width=True,
                key="ingest_btn"
            ):
                with st.spinner(
                    f"Ingesting {uploaded_file.name}...\n"
                    "Large PDFs take 3-5 minutes. Please wait."
                ):
                    result = upload_manual(
                        uploaded_file.getvalue(),
                        uploaded_file.name
                    )

                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    # Put newly uploaded manual at TOP
                    new_manual  = uploaded_file.name.replace(
                        ".pdf", ""
                    )
                    all_names   = result.get(
                        "manual_names", []
                    )
                    # New manual first then rest
                    updated = [new_manual] + [
                        m for m in all_names
                        if m != new_manual
                    ]
                    st.session_state["manual_list"] = updated
                    st.session_state["last_uploaded"] = \
                        new_manual

                    st.success(
                        f"✅ **Ingested!**\n\n"
                        f"📄 Pages: {result['pages']}\n\n"
                        f"🧩 Chunks: {result['chunks_added']}\n\n"
                        f"🗄️ Total: {result['total_chunks']}"
                    )
                    st.rerun()

        st.markdown("---")

        # ── Settings ──────────────────────────────────────
        st.markdown(
            '<div class="section-title">⚙️ Settings</div>',
            unsafe_allow_html=True
        )

        top_k_retrieve = st.slider(
            "Chunks to retrieve", 5, 50, 20
        )
        top_k_rerank = st.slider(
            "Chunks after re-rank", 1, 10, 3
        )

        st.markdown("---")

        # ── Manual Filter ─────────────────────────────────
        st.markdown(
            '<div class="section-title">📖 Filter Manual</div>',
            unsafe_allow_html=True
        )

        # Always fetch FRESH list from API — no caching
        manuals_data = get_manuals_fresh()
        api_manuals  = [
            m["manual_name"]
            for m in manuals_data.get("manuals", [])
        ]

        # Get session state list — has newest upload at top
        session_list = st.session_state.get(
            "manual_list", []
        )

        # Build final list:
        # 1. Session list first (newest upload at top)
        # 2. Then any API manuals not already in session list
        combined = list(dict.fromkeys(
            session_list + api_manuals
        ))

        # If session is empty just use API list
        if not combined:
            combined = api_manuals

        manual_names  = ["All Manuals"] + combined
        manual_filter = st.selectbox(
            "Select manual",
            manual_names,
            label_visibility="collapsed"
        )

        # Show which manual is being searched
        if manual_filter != "All Manuals":
            st.markdown(
                f'<div style="background:#0f2034;'
                f'border:1px solid #f97316;'
                f'border-radius:6px; padding:0.5rem;'
                f'font-size:0.8rem; color:#f97316;'
                f'margin-top:0.5rem;">'
                f'🎯 Searching only:<br>'
                f'<strong>{manual_filter}</strong>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#0f2034;'
                'border:1px solid #2d4a7a;'
                'border-radius:6px; padding:0.5rem;'
                'font-size:0.8rem; color:#64748b;'
                'margin-top:0.5rem;">'
                '🔍 Searching all manuals'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Recent History ────────────────────────────────
        st.markdown(
            '<div class="section-title">📜 Recent</div>',
            unsafe_allow_html=True
        )

        history = get_history()
        if history:
            for item in history[:5]:
                q    = item.get("query", "")[:40] + "..."
                conf = item.get("confidence", "")
                secs = item.get("response_time_s", 0)
                st.markdown(
                    f'<div class="history-item">'
                    f'🔍 {q}<br>'
                    f'<span style="color:#64748b;'
                    f'font-size:0.75rem;">'
                    f'{conf} | {secs:.0f}s'
                    f'</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<small style="color:#64748b;">'
                'No queries yet</small>',
                unsafe_allow_html=True
            )

    return top_k_retrieve, top_k_rerank, manual_filter


# ════════════════════════════════════════════════════════
# Main Content
# ════════════════════════════════════════════════════════

def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>🔧 Industrial Maintenance RAG</h1>
        <p>
            Describe any machine fault in plain English or
            your own language — get instant cited repair
            instructions from your official manuals.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_query_box():
    st.markdown(
        '<div class="section-title">📝 Describe the Fault</div>',
        unsafe_allow_html=True
    )

    examples = [
        "Select an example query...",
        "Motor bearing overheating after 2 hours",
        "Compressor not reaching target pressure",
        "Excessive vibration and noise from motor",
        "Oil leaking from compressor seal area",
        "Motor fails to start under load",
        "Hydraulic pump pressure dropping suddenly"
    ]

    selected = st.selectbox(
        "Examples", examples,
        label_visibility="collapsed"
    )

    default = "" if selected == "Select an example query..." \
              else selected

    query = st.text_area(
        "Fault",
        value      =default,
        height     =140,
        placeholder=(
            "Describe the fault in detail...\n\n"
            "Example: Motor on press line 3 overheating "
            "after 2 hours. Bearing temperature very high. "
            "Fault code E-042 on display panel.\n\n"
            "You can also type in Tamil, Hindi, or any language."
        ),
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        submit = st.button(
            "🔍 Get Repair Instructions",
            use_container_width=True
        )
    with col2:
        clear = st.button(
            "🗑️ Clear",
            use_container_width=True
        )
    with col3:
        st.markdown(
            '<div style="color:#64748b; font-size:0.8rem;'
            'padding-top:0.7rem; text-align:center;">'
            '30-60s</div>',
            unsafe_allow_html=True
        )

    if clear:
        st.rerun()

    return query, submit


def render_result(result: dict):
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
        return

    # ── Metrics ──────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            "⏱️ Time",
            f"{result.get('response_time_s', 0):.1f}s"
        )
    with col2:
        st.metric("📄 Chunks", result.get("chunks_used", 0))
    with col3:
        st.metric(
            "🎯 Confidence",
            result.get("confidence", "N/A")
        )
    with col4:
        st.metric("🤖 Model", result.get("model", "N/A"))
    with col5:
        st.metric(
            "🌐 Language",
            result.get("detected_language", "English")
        )

    # Show translation notice if non-English
    lang = result.get("detected_language", "English")
    if lang != "English":
        st.info(
            f"🌐 Query detected in **{lang}** — "
            f"searched manuals in English — "
            f"answer translated back to **{lang}**"
        )

    # ── Answer ───────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-title">🛠️ Repair Instructions</div>',
        unsafe_allow_html=True
    )
    answer = result.get("answer", "No answer generated")
    st.markdown(
        f'<div class="answer-card">{answer}</div>',
        unsafe_allow_html=True
    )

    # ── Sources ──────────────────────────────────────────
    sources = result.get("sources", [])
    if sources:
        st.markdown("---")
        st.markdown(
            '<div class="section-title">📚 Sources</div>',
            unsafe_allow_html=True
        )
        for s in sources:
            conf  = s.get("confidence", "Unknown")
            badge = (
                "badge-green"  if conf == "Very High" else
                "badge-blue"   if conf == "High"      else
                "badge-yellow" if conf == "Medium"    else
                "badge-red"
            )
            st.markdown(f"""
            <div class="source-card">
                <div style="display:flex;
                            justify-content:space-between;
                            align-items:center;
                            margin-bottom:0.5rem;">
                    <strong style="color:#f97316;">
                        [SOURCE {s['source_num']}]
                    </strong>
                    <span class="badge {badge}">{conf}</span>
                </div>
                <div style="color:#e2e8f0;">
                    📖 <strong>{s['manual_name']}</strong>
                </div>
                <div style="color:#94a3b8; margin-top:0.3rem;">
                    📄 Page {s['page_number']}
                    &nbsp;·&nbsp; {s['section_title']}
                </div>
                <div style="color:#475569; font-size:0.8rem;
                            margin-top:0.3rem;">
                    Score: {s['rerank_score']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Feedback ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-title">💬 Rate Answer</div>',
        unsafe_allow_html=True
    )

    query_id = result.get("query_id", "unknown")
    col1, col2 = st.columns(2)
    with col1:
        rating = st.slider(
            "Rating", 1, 5, 4,
            key=f"r_{query_id}"
        )
    with col2:
        helpful = st.radio(
            "Helpful?", ["Yes", "No"],
            key=f"h_{query_id}"
        )
        comment = st.text_input(
            "Comment",
            key=f"c_{query_id}"
        )

    if st.button(
        "Submit Feedback",
        key=f"fb_{query_id}"
    ):
        submit_feedback(
            query_id,
            rating,
            helpful == "Yes",
            comment or None
        )
        st.success("✅ Feedback submitted!")


def render_welcome():
    manuals_data  = get_manuals_fresh()
    total_chunks  = manuals_data.get("total_chunks", 0)
    total_manuals = manuals_data.get("total_manuals", 0)
    manuals_list  = manuals_data.get("manuals", [])

    if total_chunks > 0:
        st.markdown(f"""
        <div style="background:linear-gradient(
                    135deg,#0f2034,#1a2744);
                    border:1px solid #2d4a7a;
                    border-radius:12px;
                    padding:1.5rem;
                    margin-bottom:1.5rem;">
            <div style="color:#22c55e; font-size:1.1rem;
                        font-weight:600; margin-bottom:1rem;">
                ✅ System Ready
            </div>
            <div style="display:flex; gap:2rem;">
                <div>
                    <div style="color:#f97316; font-size:1.8rem;
                                font-weight:700;">
                        {total_manuals}
                    </div>
                    <div style="color:#64748b; font-size:0.8rem;">
                        Manuals
                    </div>
                </div>
                <div>
                    <div style="color:#f97316; font-size:1.8rem;
                                font-weight:700;">
                        {total_chunks:,}
                    </div>
                    <div style="color:#64748b; font-size:0.8rem;">
                        Chunks Indexed
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if manuals_list:
            st.markdown(
                '<div class="section-title">'
                '📚 Ingested Manuals</div>',
                unsafe_allow_html=True
            )
            for m in manuals_list:
                st.markdown(f"""
                <div style="background:#1e2d3d;
                            border:1px solid #2d4a7a;
                            border-radius:8px;
                            padding:0.8rem;
                            margin:0.4rem 0;
                            display:flex;
                            justify-content:space-between;">
                    <span>📖 {m['manual_name']}</span>
                    <span style="color:#f97316;">
                        {m['chunk_count']} chunks
                    </span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1e2034;
                    border:2px dashed #2d4a7a;
                    border-radius:12px;
                    padding:2rem;
                    text-align:center;">
            <div style="font-size:3rem; margin-bottom:1rem;">
                📤
            </div>
            <div style="color:#f97316; font-size:1.1rem;
                        font-weight:600; margin-bottom:0.5rem;">
                No Manuals Yet
            </div>
            <div style="color:#64748b;">
                Upload a PDF manual using the sidebar
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    # Initialize session state
    if "manual_list" not in st.session_state:
        st.session_state["manual_list"] = []
    if "last_uploaded" not in st.session_state:
        st.session_state["last_uploaded"] = None

    top_k_retrieve, top_k_rerank, manual_filter = \
        render_sidebar()

    render_hero()
    query, submit = render_query_box()

    if submit:
        if not query or len(query.strip()) < 5:
            st.warning(
                "⚠️ Please enter at least 5 characters"
            )
            return

        with st.spinner(
            "🔍 Searching manuals and generating answer... "
            "30-60 seconds on CPU"
        ):
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