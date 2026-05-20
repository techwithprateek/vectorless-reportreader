"""
app.py
------
ReportReader — Streamlit UI

Three-screen flow:
  Screen 1 (Upload):      Drag-and-drop PDF → PageIndex builds tree
  Screen 2 (Key Metrics): Auto-extracted Revenue, PAT, EPS, D/E shown as cards
  Screen 3 (Q&A):         Free-text questions with section + page citations

Session state keys used across the app:
  st.session_state.stage        → "upload" | "loaded"
  st.session_state.doc_id       → PageIndex doc_id string
  st.session_state.tree         → full tree dict (PageIndex structure)
  st.session_state.metadata     → doc metadata dict (name, page count, etc.)
  st.session_state.metrics      → list of extracted metric dicts
  st.session_state.chat_history → list of {role, content} dicts (Q&A turns)
  st.session_state.index_mgr    → IndexManager instance (reused across turns)

Why session_state for the index manager?
  Streamlit re-runs the entire script on every interaction.
  If we created a new IndexManager each time, we'd lose the cached tree.
  Storing it in session_state keeps one instance alive for the session.

Run with:
  streamlit run app.py
"""

import os
import tempfile
import logging

import streamlit as st
from dotenv import load_dotenv

from indexer import IndexManager
from retriever import answer_question, extract_key_metrics
from prompts import TREE_NODE_LABEL

# Load .env so API keys are available to LiteLLM and PageIndex
load_dotenv()

# Configure logging — debug output goes to terminal, not the Streamlit UI
logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="ReportReader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# SESSION STATE INITIALISATION
# ===========================================================================
# Streamlit reruns the script from top on every interaction, so we must
# guard every session_state key with a "not in" check to avoid resetting
# values the user has already set.
# ===========================================================================

def _init_session_state():
    """Initialise all session state keys to their default values."""

    # Which screen the user is currently on
    if "stage" not in st.session_state:
        st.session_state.stage = "upload"

    # PageIndex doc_id — set after a PDF is indexed
    if "doc_id" not in st.session_state:
        st.session_state.doc_id = None

    # Full hierarchical tree dict from PageIndex
    if "tree" not in st.session_state:
        st.session_state.tree = None

    # Document metadata (name, page count, description)
    if "metadata" not in st.session_state:
        st.session_state.metadata = None

    # Auto-extracted key metrics list (Screen 2)
    if "metrics" not in st.session_state:
        st.session_state.metrics = None

    # Q&A conversation history — list of {role, content} dicts
    # Passed to the LLM on every turn so follow-up questions work
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # IndexManager instance — stored here so it survives Streamlit reruns
    if "index_mgr" not in st.session_state:
        st.session_state.index_mgr = None

_init_session_state()


# ===========================================================================
# SIDEBAR — Document Tree Map
# ===========================================================================
# Shows the PageIndex tree as a collapsible outline in the sidebar.
# This makes the "vectorless" indexing step tangible for the user:
# they can see exactly how the document was structured.
# ===========================================================================

def _render_sidebar():
    """Render the sidebar: model info and document tree map (if loaded)."""

    with st.sidebar:
        st.title("📊 ReportReader")
        st.caption("Vectorless RAG · No embeddings · No chunking")

        st.divider()

        # --- Model info ---
        model = os.getenv("LLM_MODEL", "gpt-4o")
        st.markdown(f"**🤖 Model:** `{model}`")

        # --- Document tree map ---
        # Only shown after a document has been indexed
        if st.session_state.tree and st.session_state.metadata:
            st.divider()
            doc_name = st.session_state.metadata.get("doc_name", "Document")
            page_count = st.session_state.metadata.get("page_count", "?")

            st.markdown(f"**📄 {doc_name}**")
            st.caption(f"{page_count} pages · PageIndex tree below")

            # Render the tree recursively
            # Each node shows its title and page range
            _render_tree_node(st.session_state.tree, depth=0)

            st.divider()

            # Reset button — lets the user upload a new document
            if st.button("📂 Load new document", use_container_width=True):
                # Clear all session state and return to upload screen
                for key in ["stage", "doc_id", "tree", "metadata",
                             "metrics", "chat_history", "index_mgr"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # --- How it works note ---
        st.divider()
        with st.expander("ℹ️ How does this work?"):
            st.markdown(
                """
                **ReportReader uses PageIndex** — a vectorless RAG approach.

                Instead of chunking and embeddings:
                1. Your PDF is parsed into a **hierarchical tree index**
                   (like a smart table of contents).
                2. When you ask a question, an LLM **reasons over the tree**
                   to find the right section — just like a human expert
                   flipping to the correct chapter.
                3. Only the relevant pages are fetched and sent to the LLM.
                4. Every answer cites the exact **section + page range**.

                No vector DB. No Pinecone. No cosine similarity.
                """
            )


def _render_tree_node(node: dict, depth: int):
    """
    Recursively render a PageIndex tree node in the sidebar.

    Each node is shown as indented text with its page range.
    Child nodes are rendered under an expander to keep the sidebar tidy.

    Parameters
    ----------
    node : dict
        A PageIndex tree node with keys: title, start_index, end_index, nodes
    depth : int
        Current recursion depth (controls indentation)
    """
    if not node:
        return

    title = node.get("title", "Section")
    start = node.get("start_index", "?")
    end = node.get("end_index", "?")
    children = node.get("nodes", [])

    # Indentation — top-level nodes are bold, deeper ones are smaller
    indent = "&nbsp;" * (depth * 4)
    label = TREE_NODE_LABEL.format(title=title, start=start, end=end)

    if depth == 0:
        # Root node — show as bold header
        st.markdown(f"**{label}**")
    else:
        # Child node — show with indentation
        st.markdown(f"{indent}↳ {label}", unsafe_allow_html=True)

    # Recurse into child nodes
    for child in children:
        _render_tree_node(child, depth + 1)


# ===========================================================================
# SCREEN 1 — Upload
# ===========================================================================

def _screen_upload():
    """
    Screen 1: PDF upload and PageIndex tree building.

    - User drags/drops or picks a PDF (max 50 MB)
    - We save it to a temp file (Streamlit gives us a BytesIO buffer)
    - IndexManager.index_pdf() builds the tree (60–120 sec for large reports)
    - On completion, we transition to "loaded" stage
    """

    st.title("📊 ReportReader")
    st.subheader("Ask questions about any Indian company's annual report")
    st.markdown(
        "Upload a PDF and get instant answers with **section + page citations**. "
        "Powered by **PageIndex** — no vector database, no chunking."
    )

    st.divider()

    # --- File uploader ---
    uploaded_file = st.file_uploader(
        "Upload Annual Report (PDF, max 50 MB)",
        type=["pdf"],
        help="Supports BSE/NSE annual reports, DRHPs, investor presentations.",
    )

    if uploaded_file is not None:
        # --- Validate file size ---
        # Streamlit doesn't enforce size limits on its own
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > 50:
            st.error(f"File too large ({file_size_mb:.1f} MB). Maximum is 50 MB.")
            return

        st.info(
            f"**{uploaded_file.name}** ({file_size_mb:.1f} MB) — "
            "ready to index. Click the button below to start."
        )

        if st.button("🚀 Build PageIndex tree", type="primary", use_container_width=True):
            _run_indexing(uploaded_file)


def _run_indexing(uploaded_file):
    """
    Save the uploaded file to disk and run PageIndex tree building.

    PageIndex needs a real file path (not a BytesIO buffer) because it
    uses PyMuPDF internally which requires a path string.
    We use a named temporary file that persists for the duration of this call.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit's UploadedFile object (BytesIO-like).
    """

    # Write the uploaded bytes to a temp file that PageIndex can read
    with tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False, prefix="reportreader_"
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    logger.info("Saved upload to temp path: %s", tmp_path)

    # --- Progress messaging ---
    # Building the tree takes 60–120 seconds for large reports.
    # We set clear expectations so the user doesn't think it's frozen.
    progress_placeholder = st.empty()
    progress_placeholder.info(
        "⏳ **Building document index…** "
        "This takes about 1–2 minutes for large reports. "
        "The LLM is reading the document and building a hierarchical tree. "
        "Please wait — this only happens once per document."
    )

    try:
        with st.spinner("Indexing in progress…"):
            # Create an IndexManager (or reuse if already in session state)
            # The workspace persists the tree to disk so future loads are instant
            if st.session_state.index_mgr is None:
                workspace_path = os.getenv("PAGEINDEX_WORKSPACE", "./workspace")
                st.session_state.index_mgr = IndexManager(workspace_path=workspace_path)

            mgr = st.session_state.index_mgr

            # Build the PageIndex tree — the heavy call
            doc_id = mgr.index_pdf(tmp_path)

            # Fetch tree and metadata immediately while spinner is active
            tree = mgr.get_tree(doc_id)
            metadata = mgr.get_metadata(doc_id)

        progress_placeholder.empty()  # Clear the waiting message

        # Store everything in session state
        st.session_state.doc_id = doc_id
        st.session_state.tree = tree
        st.session_state.metadata = metadata
        st.session_state.stage = "loaded"

        # Auto-extract key metrics in the background (shown on Screen 2)
        # This is optional enrichment and should not block the loaded view.
        try:
            with st.spinner("Extracting key metrics…"):
                metrics = extract_key_metrics(mgr, doc_id, tree)
                st.session_state.metrics = metrics
        except Exception as e:
            logger.exception("Key-metrics extraction failed: %s", e)
            st.session_state.metrics = []
            st.warning(
                "⚠️ Document indexed successfully, but key metrics could not be "
                "extracted right now. Q&A is still available."
            )

        st.success("✅ Document indexed! Switching to report view…")
        st.rerun()  # Transition to the loaded layout

    except Exception as e:
        progress_placeholder.empty()
        logger.exception("Indexing failed: %s", e)
        st.error(
            f"❌ Indexing failed: {e}\n\n"
            "Check that your API key is set in `.env` and the PDF is valid."
        )
    finally:
        # Clean up the temp file regardless of success/failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info("Cleaned up temp file: %s", tmp_path)


# ===========================================================================
# SCREEN 2 — Key Metrics
# ===========================================================================

def _screen_metrics():
    """
    Screen 2: Auto-extracted financial headline metrics.

    Shows 4 metric cards (Revenue, PAT, EPS, D/E Ratio).
    Each card displays the value and a citation chip (section + page).

    Metrics are extracted once (during indexing) and cached in session state.
    """

    st.subheader("📈 Key Financial Metrics")
    st.caption(
        "Auto-extracted using PageIndex. Each value cites the exact section and page."
    )

    metrics = st.session_state.metrics

    if not metrics:
        st.warning(
            "Could not extract key metrics automatically. "
            "Try asking in the Q&A tab: 'What is the revenue?' "
            "or 'What is the profit after tax?'"
        )
        return

    # Render metrics in a 2-column grid
    # Each card shows: metric name, value (big text), and citation
    cols = st.columns(2)
    for i, metric in enumerate(metrics):
        col = cols[i % 2]
        with col:
            _render_metric_card(metric)


def _render_metric_card(metric: dict):
    """
    Render a single metric as a styled card using Streamlit's metric widget.

    Parameters
    ----------
    metric : dict
        {
          "metric":     "Revenue from Operations",
          "value":      "₹9,14,855 Cr",   (or null if not found)
          "section":    "Financial Highlights",
          "page_range": "33-35"
        }
    """
    name = metric.get("metric", "Unknown")
    value = metric.get("value") or "Not found"
    section = metric.get("section", "")
    page_range = metric.get("page_range", "")

    with st.container(border=True):
        # Large metric value using st.metric
        st.metric(label=name, value=value if value != "null" else "—")

        # Citation chip below the value
        if section and page_range and value not in ("Not found", "null", None):
            st.caption(f"📍 {section} · p.{page_range}")


# ===========================================================================
# SCREEN 3 — Q&A
# ===========================================================================

def _screen_qa():
    """
    Screen 3: Conversational Q&A with section + page citations.

    Flow per question:
      1. User types a question (or clicks a suggestion chip)
      2. retriever.answer_question() runs two-step retrieval:
         a. LLM navigates tree → identifies sections + page ranges
         b. PageIndex fetches those pages' text
         c. LLM generates answer with inline citations
      3. Answer is appended to chat_history and displayed

    Conversation history is preserved in session state across Streamlit reruns
    so follow-up questions work correctly.
    """

    st.subheader("💬 Ask about this report")
    st.caption(
        "Every answer cites the exact section and page. "
        "The LLM only reads the pages it needs — not the whole document."
    )

    # --- Suggested question chips ---
    # These help users get started without a blank-page problem
    suggestions = [
        "What are the key risk factors?",
        "What did management say about future outlook?",
        "What is the dividend declared?",
        "How did revenue change vs last year?",
        "What is the promoter holding percentage?",
    ]

    st.markdown("**Suggested questions:**")
    # Render suggestions as clickable buttons in a horizontal row
    # We use columns so they appear inline
    suggestion_cols = st.columns(len(suggestions))
    clicked_suggestion = None
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                clicked_suggestion = suggestion

    st.divider()

    # --- Render conversation history ---
    # Iterate through past turns and display them as a chat thread.
    # We separate user messages and assistant responses visually.
    for turn in st.session_state.chat_history:
        role = turn["role"]
        content = turn["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="📊"):
                # The content may include citations embedded in the answer text.
                # We render as markdown so **bold** citations display correctly.
                st.markdown(content)

    # --- Handle new question ---
    # Determine the question source: typed input OR clicked suggestion chip
    user_input = st.chat_input("Ask anything about this report…")
    question = clicked_suggestion or user_input

    if question:
        _handle_question(question)


def _handle_question(question: str):
    """
    Process a user question through the two-step retrieval pipeline.

    1. Show the question in the chat thread immediately
    2. Run tree navigation + page fetch + answer generation (with spinner)
    3. Display the answer with a citation block below it
    4. Append both turns to chat_history for follow-up context

    Parameters
    ----------
    question : str
        The user's question (typed or from a suggestion chip).
    """

    # Show the user's message immediately (before the LLM responds)
    with st.chat_message("user"):
        st.markdown(question)

    # Append user turn to history BEFORE the LLM call so it's included
    # in the conversation context passed to the LLM
    st.session_state.chat_history.append({"role": "user", "content": question})

    # --- Run the two-step retrieval ---
    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("Navigating document tree…"):
            try:
                answer, citations = answer_question(
                    index_manager=st.session_state.index_mgr,
                    doc_id=st.session_state.doc_id,
                    question=question,
                    chat_history=st.session_state.chat_history[:-1],  # exclude current user msg
                    tree=st.session_state.tree,
                )
            except Exception as e:
                logger.exception("Q&A failed: %s", e)
                answer = (
                    f"❌ An error occurred: {e}\n\n"
                    "Check your API key and try again."
                )
                citations = []

        # --- Display the answer ---
        st.markdown(answer)

        # --- Display citation chips below the answer ---
        # These are separate from the inline citations in the answer text —
        # they give the user a quick visual summary of which pages were read.
        if citations:
            st.markdown("**📍 Retrieved from:**")
            citation_cols = st.columns(len(citations))
            for i, cite in enumerate(citations):
                with citation_cols[i]:
                    st.info(
                        f"**{cite['section_title']}**\n\np. {cite['page_range']}"
                    )

    # Append assistant response to history for future follow-up context
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ===========================================================================
# MAIN LAYOUT
# ===========================================================================

def main():
    """
    Main entry point — decides which screen to render based on session state.

    Layout:
      - Sidebar (always): model info + document tree map (when loaded)
      - Main area:
          - "upload" stage → Screen 1 (upload + indexing)
          - "loaded" stage → tabs for Screen 2 (metrics) + Screen 3 (Q&A)
    """

    # Always render the sidebar regardless of current screen
    _render_sidebar()

    if st.session_state.stage == "upload":
        # Screen 1: Upload and index
        _screen_upload()

    elif st.session_state.stage == "loaded":
        # Document is loaded — show metadata header and tabs
        meta = st.session_state.metadata or {}
        doc_name = meta.get("doc_name", "Report")
        page_count = meta.get("page_count", "?")
        description = meta.get("description", "")

        # --- Document header ---
        st.title(f"📄 {doc_name}")
        if description:
            st.caption(description)
        st.caption(f"{page_count} pages · Indexed with PageIndex")

        st.divider()

        # --- Tabs for Screen 2 and Screen 3 ---
        tab_metrics, tab_qa = st.tabs(["📈 Key Metrics", "💬 Q&A"])

        with tab_metrics:
            _screen_metrics()

        with tab_qa:
            _screen_qa()


# Run the app
if __name__ == "__main__":
    main()
