import json
import streamlit as st
from qna import index_document, ask, get_metadata, get_chunks, compare_documents, store

st.set_page_config(page_title="FinDoc Analyser v2", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .block-container { padding: 1.5rem 2rem 2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }
    [data-testid="stSidebar"] .stButton button {
        background: #161b22;
        border: 1px solid #30363d;
        color: #e6edf3 !important;
        border-radius: 6px;
        font-family: 'Sora', sans-serif;
        font-size: 0.82rem;
    }
    [data-testid="stSidebar"] .stButton button:hover { border-color: #58a6ff; }

    /* Primary button */
    .stButton button[kind="primary"] {
        background: #1f6feb !important;
        border: none !important;
        color: white !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #21262d;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Sora', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.6rem 1.4rem;
        color: #8b949e;
        border-bottom: 2px solid transparent;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
        background: transparent !important;
    }

    /* Cards */
    .fin-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .fin-card h4 { margin: 0 0 0.5rem; color: #e6edf3; font-size: 1rem; font-weight: 600; }
    .fin-card p  { margin: 0; color: #8b949e; font-size: 0.85rem; line-height: 1.6; }

    /* KV grid */
    .kv-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.6rem; margin-top: 0.8rem; }
    .kv-item {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
    }
    .kv-label { font-size: 0.7rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.04em; }
    .kv-value { font-size: 0.95rem; font-weight: 600; color: #3fb950; font-family: 'DM Mono', monospace; }
    .kv-null  { font-size: 0.95rem; color: #484f58; font-family: 'DM Mono', monospace; }

    /* Chunk pill */
    .section-pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        background: #1f3a5f;
        color: #58a6ff;
        margin-bottom: 0.4rem;
    }
    .score-bar-wrap { display: flex; align-items: center; gap: 0.6rem; margin: 0.3rem 0; }
    .score-bar-bg { flex: 1; height: 6px; background: #21262d; border-radius: 3px; }
    .score-bar-fill { height: 6px; border-radius: 3px; background: linear-gradient(90deg, #1f6feb, #3fb950); }
    .score-label { font-size: 0.75rem; font-family: 'DM Mono', monospace; color: #3fb950; min-width: 42px; }

    /* Chat */
    .stChatMessage { border-radius: 10px; }
    .chunk-used {
        background: #0d1117;
        border-left: 3px solid #1f6feb;
        border-radius: 0 6px 6px 0;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
        color: #8b949e;
        font-family: 'DM Mono', monospace;
    }
    .chunk-used .ch-meta { color: #58a6ff; font-weight: 500; margin-bottom: 0.2rem; }

    /* Comparison */
    .comp-col-header {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px 8px 0 0;
        padding: 0.7rem 1rem;
        font-weight: 600;
        color: #e6edf3;
        font-size: 0.9rem;
    }
    .comp-answer {
        background: #0d1117;
        border: 1px solid #21262d;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
        font-size: 0.88rem;
        color: #c9d1d9;
        min-height: 120px;
    }
    .tag { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.72rem; font-weight: 600; margin-right: 0.3rem; }
    .tag-10k  { background: #1a3a2a; color: #3fb950; }
    .tag-10q  { background: #1a2a3a; color: #58a6ff; }
    .tag-other { background: #2a2a1a; color: #d29922; }
</style>
""", unsafe_allow_html=True)

# ── session state ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 FinDoc v2")
    st.caption("SEC filing analysis & comparison")
    st.divider()

    uploaded_file = st.file_uploader("Upload a 10-K / 10-Q PDF", type=["pdf"])
    if uploaded_file:
        if st.button("Index Document", use_container_width=True, type="primary"):
            with st.spinner("Indexing & extracting metadata…"):
                msg = index_document(uploaded_file)
                st.session_state.active_doc = uploaded_file.name
                st.session_state.chat_history = []
            st.success(msg)

    st.divider()

    docs = store.list_documents()
    all_meta = store.list_metadata()

    if docs:
        st.markdown("**Indexed documents**")
        for doc in docs:
            meta = all_meta.get(doc, {})
            company = meta.get("company_name") or doc
            doc_type = meta.get("doc_type") or ""
            year = meta.get("fiscal_year") or ""
            label = f"{company}"
            if year:
                label += f" ({year})"
            is_active = st.session_state.active_doc == doc
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f"{'▶ ' if is_active else ''}{label}", key=f"sel_{doc}", use_container_width=True):
                    st.session_state.active_doc = doc
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                if st.button("✕", key=f"del_{doc}"):
                    store.delete_document(doc)
                    if st.session_state.active_doc == doc:
                        st.session_state.active_doc = None
                        st.session_state.chat_history = []
                    st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── main ───────────────────────────────────────────────────────────────────────
tab_chat, tab_docs, tab_chunks, tab_compare = st.tabs(
    ["💬 Chat", "📁 Documents", "🔍 Chunk Viewer", "⚖️ Comparison"]
)

# ─── TAB 1: CHAT ──────────────────────────────────────────────────────────────
with tab_chat:
    if st.session_state.active_doc:
        meta = store.load_metadata(st.session_state.active_doc) or {}
        company = meta.get("company_name") or st.session_state.active_doc
        st.markdown(f"**Active:** {company} &nbsp;·&nbsp; `{st.session_state.active_doc}`")
    else:
        st.info("Select or upload a document from the sidebar.")

    for role, message, chunks in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
            if role == "assistant" and chunks:
                with st.expander(f"Chunks used ({len(chunks)}) — scores", expanded=False):
                    for c in chunks:
                        pct = int(c["score"] * 100)
                        section = c.get("section_hint") or "—"
                        st.markdown(
                            f'<div class="chunk-used">'
                            f'<div class="ch-meta">Chunk #{c["chunk_index"]} &nbsp;·&nbsp; {section} &nbsp;·&nbsp; score: {c["score"]:.4f}</div>'
                            f'<div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
                            f'<span class="score-label">{pct}%</span></div>'
                            f'{c["text"][:300]}{"…" if len(c["text"]) > 300 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    query = st.chat_input(
        "Ask a question about the document…",
        disabled=not st.session_state.active_doc,
    )

    if query:
        st.session_state.chat_history.append(("user", query, []))
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving & generating…"):
                answer, scored_chunks = ask(query, doc_name=st.session_state.active_doc)
            st.markdown(answer)
            if scored_chunks:
                with st.expander(f"Chunks used ({len(scored_chunks)}) — scores", expanded=True):
                    for c in scored_chunks:
                        pct = int(c["score"] * 100)
                        section = c.get("section_hint") or "—"
                        st.markdown(
                            f'<div class="chunk-used">'
                            f'<div class="ch-meta">Chunk #{c["chunk_index"]} &nbsp;·&nbsp; {section} &nbsp;·&nbsp; score: {c["score"]:.4f}</div>'
                            f'<div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
                            f'<span class="score-label">{pct}%</span></div>'
                            f'{c["text"][:300]}{"…" if len(c["text"]) > 300 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.chat_history.append(("assistant", answer, scored_chunks))


# ─── TAB 2: DOCUMENTS ─────────────────────────────────────────────────────────
with tab_docs:
    docs = store.list_documents()
    all_meta = store.list_metadata()

    if not docs:
        st.info("No documents indexed yet. Upload a PDF from the sidebar.")
    else:
        for doc in docs:
            meta = all_meta.get(doc, {})
            company = meta.get("company_name") or doc
            doc_type = (meta.get("doc_type") or "").upper()
            year = meta.get("fiscal_year") or ""
            summary = meta.get("summary") or ""
            financials = meta.get("financials") or {}
            chunk_count = meta.get("chunk_count", "?")

            tag_cls = "tag-10k" if "10-K" in doc_type else ("tag-10q" if "10-Q" in doc_type else "tag-other")

            st.markdown(
                f'<div class="fin-card">'
                f'<h4>{company}'
                f'  <span class="tag {tag_cls}">{doc_type or "DOC"}</span>'
                f'  {"<span class=\\"tag tag-other\\">" + year + "</span>" if year else ""}'
                f'</h4>'
                f'<p style="color:#8b949e;font-size:0.75rem;margin-bottom:0.5rem">'
                f'📄 {doc} &nbsp;·&nbsp; {chunk_count} chunks</p>'
                f'<p>{summary}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if financials:
                kv_html = '<div class="kv-grid">'
                for key, val in financials.items():
                    label = key.replace("_", " ").title()
                    if val:
                        kv_html += f'<div class="kv-item"><div class="kv-label">{label}</div><div class="kv-value">{val}</div></div>'
                    else:
                        kv_html += f'<div class="kv-item"><div class="kv-label">{label}</div><div class="kv-null">—</div></div>'
                kv_html += '</div>'
                st.markdown(kv_html, unsafe_allow_html=True)

            col_dl, _ = st.columns([1, 5])
            with col_dl:
                st.download_button(
                    label="⬇ Download JSON",
                    data=json.dumps(meta, indent=2),
                    file_name=f"{doc.replace('.pdf', '')}_metadata.json",
                    mime="application/json",
                    key=f"dl_{doc}",
                )
            st.markdown("<hr style='border-color:#21262d;margin:1.5rem 0'>", unsafe_allow_html=True)


# ─── TAB 3: CHUNK VIEWER ──────────────────────────────────────────────────────
with tab_chunks:
    docs = store.list_documents()
    if not docs:
        st.info("No documents indexed yet.")
    else:
        all_meta = store.list_metadata()
        doc_labels = {d: (all_meta.get(d, {}).get("company_name") or d) for d in docs}
        selected = st.selectbox(
            "Select document",
            options=docs,
            format_func=lambda d: f"{doc_labels[d]}  ({d})",
        )

        if selected:
            chunks = get_chunks(selected)
            st.markdown(f"**{len(chunks)} chunks** for `{selected}`")

            section_options = ["All"] + sorted({c.section_hint for c in chunks if c.section_hint})
            filter_section = st.selectbox("Filter by section", section_options, key="chunk_section_filter")

            display_chunks = chunks if filter_section == "All" else [c for c in chunks if c.section_hint == filter_section]
            st.caption(f"Showing {len(display_chunks)} chunks")

            for c in display_chunks:
                section = c.section_hint or "Unclassified"
                chars = f"{c.char_start}–{c.char_end}"
                with st.expander(f"Chunk #{c.chunk_index}  ·  {section}  ·  chars {chars}", expanded=False):
                    st.markdown(
                        f'<div class="section-pill">{section}</div>',
                        unsafe_allow_html=True,
                    )
                    st.text(c.text)


# ─── TAB 4: COMPARISON ────────────────────────────────────────────────────────
with tab_compare:
    docs = store.list_documents()
    if len(docs) < 2:
        st.info("Index at least 2 documents to use comparison.")
    else:
        all_meta = store.list_metadata()
        doc_labels = {d: (all_meta.get(d, {}).get("company_name") or d) for d in docs}

        selected_docs = st.multiselect(
            "Select documents to compare (2 or more)",
            options=docs,
            format_func=lambda d: f"{doc_labels[d]}  ({d})",
            default=docs[:2],
        )

        comp_query = st.text_input("Comparison question", placeholder="e.g. What are the main risk factors?")

        if st.button("Run Comparison", type="primary", disabled=len(selected_docs) < 2 or not comp_query):
            with st.spinner("Running comparison across documents…"):
                results = compare_documents(comp_query, selected_docs)

            cols = st.columns(len(selected_docs))
            for col, doc in zip(cols, selected_docs):
                label = doc_labels[doc]
                answer, scored_chunks = results[doc]
                with col:
                    st.markdown(f'<div class="comp-col-header">{label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="comp-answer">{answer}</div>', unsafe_allow_html=True)
                    if scored_chunks:
                        with st.expander(f"Chunks ({len(scored_chunks)})", expanded=False):
                            for c in scored_chunks:
                                pct = int(c["score"] * 100)
                                st.markdown(
                                    f'<div class="chunk-used">'
                                    f'<div class="ch-meta">#{c["chunk_index"]} · {c.get("section_hint") or "—"} · {c["score"]:.4f}</div>'
                                    f'<div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
                                    f'<span class="score-label">{pct}%</span></div>'
                                    f'{c["text"][:200]}…</div>',
                                    unsafe_allow_html=True,
                                )

        # Side-by-side financial metrics comparison
        if len(selected_docs) >= 2:
            st.markdown("---")
            st.markdown("#### Financial Metrics Comparison")
            fin_keys = [
                "revenue", "net_income", "eps_diluted", "gross_margin", "operating_margin",
                "net_margin", "total_assets", "total_liabilities", "shareholders_equity",
                "debt_to_equity", "current_ratio", "return_on_equity", "free_cash_flow",
            ]
            header_cols = st.columns([2] + [2] * len(selected_docs))
            header_cols[0].markdown("**Metric**")
            for i, doc in enumerate(selected_docs):
                header_cols[i + 1].markdown(f"**{doc_labels[doc]}**")

            for key in fin_keys:
                row_cols = st.columns([2] + [2] * len(selected_docs))
                row_cols[0].markdown(f"<span style='color:#8b949e;font-size:0.85rem'>{key.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
                for i, doc in enumerate(selected_docs):
                    meta = all_meta.get(doc, {})
                    val = (meta.get("financials") or {}).get(key)
                    row_cols[i + 1].markdown(
                        f"<span style='font-family:DM Mono,monospace;font-size:0.9rem;color:{'#3fb950' if val else '#484f58'}'>{val or '—'}</span>",
                        unsafe_allow_html=True,
                    )
