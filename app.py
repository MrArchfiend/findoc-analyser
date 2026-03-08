import streamlit as st
from qna import index_document, ask, store

# --- Page config ---
st.set_page_config(
    page_title="FinDoc Analyser",
    page_icon="📊",
    layout="wide",
)

# --- Minimal styling ---
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; }
        .stChatMessage { border-radius: 10px; }
        .status-box {
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of (role, message) tuples
if "indexed_doc" not in st.session_state:
    st.session_state.indexed_doc = None


# --- Sidebar ---
with st.sidebar:
    st.title("📊 FinDoc Analyser")
    st.caption("Ask questions about any 10-K / SEC filing")
    st.divider()

    # File upload
    uploaded_file = st.file_uploader("Upload a 10-K PDF", type=["pdf"])

    if uploaded_file:
        if st.button("Index Document", use_container_width=True, type="primary"):
            with st.spinner("Indexing document..."):
                msg = index_document(uploaded_file)
                st.session_state.indexed_doc = uploaded_file.name
                st.session_state.chat_history = []   # reset chat for new doc
            st.success(msg)

    st.divider()

    # Show already indexed documents
    docs = store.list_documents()
    if docs:
        st.markdown("**Indexed documents**")
        selected = st.selectbox(
            "Switch document",
            options=docs,
            index=docs.index(st.session_state.indexed_doc)
                   if st.session_state.indexed_doc in docs else 0,
            label_visibility="collapsed",
        )
        st.session_state.indexed_doc = selected

        if st.button("🗑 Remove document", use_container_width=True):
            store.delete_document(st.session_state.indexed_doc)
            st.session_state.indexed_doc = None
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# --- Main area ---
st.header("Financial Document Q&A")

# Show active document
if st.session_state.indexed_doc:
    st.markdown(
        f'<div class="status-box" style="background:#e8f4ea;color:#1a6b2e;">'
        f'📄 Active document: <strong>{st.session_state.indexed_doc}</strong></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-box" style="background:#fff3cd;color:#856404;">'
        '⚠️ No document selected. Upload and index a PDF from the sidebar.</div>',
        unsafe_allow_html=True,
    )

# Render chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# Chat input
query = st.chat_input(
    "Ask a question about the document...",
    disabled=not st.session_state.indexed_doc,
)

if query:
    # Show user message
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and show answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(query, doc_name=st.session_state.indexed_doc)
        st.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))
