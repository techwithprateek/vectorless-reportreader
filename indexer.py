"""
indexer.py
----------
Wraps the PageIndex self-hosted client for ReportReader.

Responsibilities:
  1. Accept a PDF path and build a hierarchical tree index using PageIndex.
  2. Persist the index in a local workspace directory so it survives
     Streamlit reruns (the tree only needs to be built once per document).
  3. Expose helpers to fetch the tree structure and document metadata.

How PageIndex self-hosted works (no vector DB required):
  - PageIndexClient reads the PDF page by page using PyMuPDF.
  - It calls an LLM (via LiteLLM) to summarise sections and build a tree —
    essentially a smart, LLM-aware table of contents.
  - The resulting tree is stored in `workspace/` as JSON files.
  - On subsequent loads, the cached tree is returned instantly.

Dependencies: pageindex, pymupdf, python-dotenv
"""

import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

# PageIndex self-hosted client — no API key needed for PageIndex itself.
# It uses your LLM API key (OpenAI / Anthropic / Ollama) for the indexing step.
from pageindex import PageIndexClient

# Load environment variables from .env so callers don't have to do it
load_dotenv()

# Module-level logger — keeps noise out of the Streamlit UI
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IndexManager
# ---------------------------------------------------------------------------
# A thin wrapper around PageIndexClient that:
#   - Creates the workspace directory if it doesn't exist
#   - Deduplicates documents so we never re-index the same PDF twice
#   - Exposes clean methods for the rest of the app to call
# ---------------------------------------------------------------------------
class IndexManager:
    """
    Manages the PageIndex tree for a single document session.

    Usage:
        manager = IndexManager(workspace_path="./workspace")
        doc_id  = manager.index_pdf("/path/to/report.pdf")
        tree    = manager.get_tree(doc_id)
        meta    = manager.get_metadata(doc_id)
    """

    def __init__(self, workspace_path: str = "./workspace"):
        """
        Initialise the IndexManager.

        Parameters
        ----------
        workspace_path : str
            Directory where PageIndex stores its cached tree JSON files.
            Created automatically if it doesn't exist.
        """
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(parents=True, exist_ok=True)

        # PageIndexClient in self-hosted mode.
        # It reads LLM credentials from environment variables:
        #   OPENAI_API_KEY     → for gpt-* models
        #   ANTHROPIC_API_KEY  → for claude-* models
        #   OLLAMA_BASE_URL    → for ollama/* models (LiteLLM handles routing)
        # The `model` parameter tells PageIndex which LLM to use for building
        # the tree (summarising sections, detecting headings, etc.).
        model = os.getenv("LLM_MODEL", "gpt-4o")
        self.client = PageIndexClient(workspace=str(self.workspace), model=model)

        logger.info("IndexManager ready. Workspace: %s | Model: %s", self.workspace, model)

    # ------------------------------------------------------------------
    # index_pdf
    # ------------------------------------------------------------------
    def index_pdf(self, pdf_path: str) -> str:
        """
        Build (or load from cache) a PageIndex tree for the given PDF.

        How it works:
          - PageIndex reads the PDF using PyMuPDF.
          - It calls the configured LLM to summarise each section and
            construct a hierarchical tree (like a table of contents).
          - The tree is persisted in `workspace/` as JSON.
          - If a tree for this file already exists, it is returned immediately
            without re-indexing (caching is handled by PageIndexClient).

        Parameters
        ----------
        pdf_path : str
            Absolute or relative path to the PDF file.

        Returns
        -------
        str
            A unique doc_id string used to reference this document in
            subsequent get_tree / get_page_content calls.
        """
        pdf_path = str(pdf_path)
        pdf_name = Path(pdf_path).name

        logger.info("Indexing PDF: %s", pdf_name)

        # client.index() is the heavy call — it may take 60–120 seconds for
        # a large report because it makes multiple LLM calls to summarise sections.
        # PageIndexClient caches by filename so re-runs are instant.
        doc_id = self.client.index(pdf_path)

        logger.info("Indexing complete. doc_id: %s", doc_id)
        return doc_id

    # ------------------------------------------------------------------
    # get_tree
    # ------------------------------------------------------------------
    def get_tree(self, doc_id: str) -> dict:
        """
        Return the full hierarchical tree structure for a document.

        The tree is a nested dict of nodes, each with:
          {
            "title":       "Financial Highlights",
            "start_index": 18,   # first page of this section (0-based)
            "end_index":   24,   # last page of this section (inclusive)
            "summary":     "Key financial metrics for FY2024...",
            "nodes":       [...]  # child nodes (sub-sections)
          }

        This tree is what the LLM navigates in retriever.py to find
        the right page range for a user's question.

        Parameters
        ----------
        doc_id : str
            The doc_id returned by index_pdf().

        Returns
        -------
        dict
            Parsed tree structure (JSON → Python dict).
        """
        # get_document_structure returns a JSON string — we parse it here
        # so the rest of the app works with plain Python dicts
        raw = self.client.get_document_structure(doc_id)
        return json.loads(raw)

    # ------------------------------------------------------------------
    # get_metadata
    # ------------------------------------------------------------------
    def get_metadata(self, doc_id: str) -> dict:
        """
        Return document metadata detected during indexing.

        Typical fields:
          {
            "doc_name":    "reliance_fy24.pdf",
            "page_count":  358,
            "description": "Reliance Industries FY2024 Annual Report",
            "status":      "indexed"
          }

        Used by the UI to display the company name and page count
        after a document has been successfully indexed.

        Parameters
        ----------
        doc_id : str
            The doc_id returned by index_pdf().

        Returns
        -------
        dict
            Parsed metadata (JSON → Python dict).
        """
        raw = self.client.get_document(doc_id)
        return json.loads(raw)

    # ------------------------------------------------------------------
    # get_page_content
    # ------------------------------------------------------------------
    def get_page_content(self, doc_id: str, page_range: str) -> str:
        """
        Fetch the raw text from a specific page range of the document.

        page_range formats accepted by PageIndex:
          "33"      → single page 33
          "33-36"   → pages 33 through 36 (inclusive)
          "33,35"   → pages 33 and 35 only

        This is called in retriever.py AFTER the tree navigation step
        has identified which pages are relevant. Only the relevant pages
        are fetched — never the full document.

        Parameters
        ----------
        doc_id : str
            The doc_id returned by index_pdf().
        page_range : str
            Page range string (see formats above).

        Returns
        -------
        str
            Raw text extracted from those pages.
        """
        return self.client.get_page_content(doc_id, page_range)
