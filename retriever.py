"""
retriever.py
------------
Two-step vectorless retrieval pipeline for ReportReader.

Step 1 — Tree Navigation (WHERE to look):
    The LLM receives the full PageIndex tree (the "smart table of contents")
    and reasons about which section(s) most likely contain the answer.
    Returns a section title + page range — no similarity search involved.

Step 2 — Answer Generation (WHAT the answer is):
    We fetch only the relevant pages identified in Step 1.
    The LLM reads that text and generates a grounded answer with citations.

Why two steps?
    This mirrors how a human expert reads a 300-page report:
      1. Flip to the index / table of contents → find the right chapter.
      2. Read that chapter → extract the answer.
    A single-step approach (ask one big question) would require sending the
    entire document to the LLM, which is expensive and slow.

LLM routing:
    All LLM calls go through LiteLLM, which transparently supports:
      - OpenAI    (LLM_MODEL=gpt-4o)
      - Anthropic (LLM_MODEL=claude-3-5-sonnet-20241022)
      - Ollama    (LLM_MODEL=ollama/llama3, OLLAMA_BASE_URL=http://localhost:11434)
      - Any other LiteLLM-supported provider

Dependencies: litellm, python-dotenv
"""

import os
import json
import logging
from typing import Optional

import litellm
from dotenv import load_dotenv

from prompts import TREE_NAVIGATION_PROMPT, KEY_METRICS_PROMPT, QA_SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)

# Suppress LiteLLM's verbose HTTP logs in the Streamlit UI
litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# _get_model
# ---------------------------------------------------------------------------
# Single source of truth for which LLM model to use.
# Reads from the LLM_MODEL env var; falls back to gpt-4o.
# ---------------------------------------------------------------------------
def _get_model() -> str:
    """Return the configured LLM model name from environment."""
    return os.getenv("LLM_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# _call_llm
# ---------------------------------------------------------------------------
# Thin wrapper around litellm.completion() that handles:
#   - Ollama base URL injection (needed for local models)
#   - Consistent error logging
#   - Returns just the message content string
# ---------------------------------------------------------------------------
def _call_llm(messages: list, model: Optional[str] = None) -> str:
    """
    Call the configured LLM via LiteLLM and return the response text.

    LiteLLM routes to the right provider based on the model name prefix:
      "gpt-*"             → OpenAI
      "claude-*"          → Anthropic
      "ollama/*"          → Ollama local server
      "bedrock/*"         → AWS Bedrock
      etc.

    For Ollama, we inject the OLLAMA_BASE_URL as the api_base so LiteLLM
    knows where the local server is running.

    Parameters
    ----------
    messages : list
        Standard OpenAI-format message list:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    model : str, optional
        Override the default model. Falls back to LLM_MODEL env var.

    Returns
    -------
    str
        The assistant's response text (stripped of leading/trailing whitespace).
    """
    model = model or _get_model()

    # LiteLLM extras — only applies when using Ollama
    kwargs = {}
    if model.startswith("ollama/"):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        kwargs["api_base"] = base_url
        logger.info("Using Ollama at %s with model %s", base_url, model)

    try:
        response = litellm.completion(model=model, messages=messages, **kwargs)
    except Exception:
        logger.exception(
            "LLM completion failed for model=%s api_base=%s",
            model,
            kwargs.get("api_base"),
        )
        raise

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# navigate_tree
# ---------------------------------------------------------------------------
# Step 1: LLM reasons over the document tree to find relevant sections.
#
# Input:  tree (the full PageIndex tree dict) + user question
# Output: list of dicts, each with section_title, start_page, end_page
#
# The LLM returns JSON — we parse it here and return Python dicts.
# If the JSON parse fails (e.g., model adds markdown fences), we fall back
# to a safe default that returns the first 10 pages.
# ---------------------------------------------------------------------------
def navigate_tree(tree: dict, question: str) -> list[dict]:
    """
    Ask the LLM to identify the most relevant sections in the tree.

    This is the "reasoning" step — the LLM reads the tree like a human
    would scan a table of contents to decide which chapter to read.

    Parameters
    ----------
    tree : dict
        The full PageIndex tree returned by IndexManager.get_tree().
    question : str
        The user's question.

    Returns
    -------
    list[dict]
        1 or 2 dicts, each with:
          {
            "section_title": str,
            "start_page":    int,
            "end_page":      int,
            "reasoning":     str   ← why this section is relevant
          }
    """
    # Serialise the tree as indented JSON so the LLM can read it easily.
    # We intentionally include summaries — they help the LLM reason without
    # having to read the actual page text yet.
    tree_str = json.dumps(tree, indent=2)

    messages = [
        {"role": "system", "content": TREE_NAVIGATION_PROMPT},
        {
            "role": "user",
            "content": (
                f"Document tree:\n{tree_str}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    logger.info("Tree navigation — question: %s", question[:80])
    raw = _call_llm(messages)

    # --- Parse the JSON response ---
    # The model sometimes wraps output in markdown fences (```json ... ```).
    # We strip those before parsing.
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        sections = json.loads(cleaned)
        logger.info("Tree navigation found %d section(s)", len(sections))
        return sections
    except json.JSONDecodeError:
        # Fall back: if we can't parse the LLM's response, return the first
        # node in the tree so the app doesn't crash
        logger.warning("Tree navigation JSON parse failed. Using fallback (first node).")
        first_node = tree.get("nodes", [{}])[0] if tree.get("nodes") else tree
        return [
            {
                "section_title": first_node.get("title", "Document"),
                "start_page": first_node.get("start_index", 1),
                "end_page": first_node.get("end_index", 10),
                "reasoning": "Fallback: could not parse tree navigation response.",
            }
        ]


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------
# Step 2: Fetch the relevant page text and generate a grounded answer.
#
# Flow:
#   1. navigate_tree() → identified sections with page ranges
#   2. For each section, fetch page text via IndexManager.get_page_content()
#   3. Concatenate all fetched text as context
#   4. Call LLM with system prompt + conversation history + context + question
#   5. Return (answer_text, citations_list)
# ---------------------------------------------------------------------------
def answer_question(
    index_manager,
    doc_id: str,
    question: str,
    chat_history: list[dict],
    tree: dict,
) -> tuple[str, list[dict]]:
    """
    Full two-step retrieval: navigate tree → fetch pages → generate answer.

    Parameters
    ----------
    index_manager : IndexManager
        The IndexManager instance (holds the PageIndexClient).
    doc_id : str
        The doc_id for the current document.
    question : str
        The user's question.
    chat_history : list[dict]
        Prior conversation turns as OpenAI-format messages
        [{"role": "user"/"assistant", "content": "..."}, ...].
        Enables follow-up questions that reference earlier answers.
    tree : dict
        The full PageIndex tree (so we don't re-fetch it every call).

    Returns
    -------
    tuple[str, list[dict]]
        - answer   : The LLM's answer text (with inline citations).
        - citations: List of dicts [{section_title, start_page, end_page}]
                     so the UI can render citation chips separately.
    """

    # --- Step 1: Navigate the tree to find relevant sections ---
    sections = navigate_tree(tree, question)

    # --- Step 2: Fetch page content for each identified section ---
    context_blocks = []
    citations = []

    for section in sections:
        title = section.get("section_title", "Unknown Section")
        start = section.get("start_page", 1)
        end = section.get("end_page", start)

        # Build the page range string expected by PageIndex (e.g. "33-36")
        page_range = f"{start}-{end}" if start != end else str(start)

        logger.info("Fetching pages %s from section: %s", page_range, title)

        # Fetch only these pages — NOT the entire document
        page_text = index_manager.get_page_content(doc_id, page_range)

        # Wrap the fetched text with its citation header so the LLM knows
        # exactly where this text came from when it writes citations
        context_blocks.append(
            f"--- [{title}, p.{page_range}] ---\n{page_text}"
        )

        citations.append({
            "section_title": title,
            "start_page": start,
            "end_page": end,
            "page_range": page_range,
        })

    # Combine all fetched sections into one context string
    full_context = "\n\n".join(context_blocks)

    # --- Step 3: Build the message list for the LLM ---
    # Structure: system prompt → conversation history → current question with context
    # We pass conversation history BEFORE the current question so the model
    # can answer follow-up questions coherently.
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        *chat_history,   # previous turns (user + assistant messages)
        {
            "role": "user",
            "content": (
                f"Retrieved context:\n\n{full_context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    logger.info("Generating answer for: %s", question[:80])
    answer = _call_llm(messages)

    return answer, citations


# ---------------------------------------------------------------------------
# extract_key_metrics
# ---------------------------------------------------------------------------
# Auto-extracts the 4 headline financial metrics shown on Screen 2.
#
# Strategy:
#   1. Ask the LLM to navigate the tree for "financial highlights / summary"
#   2. Fetch those pages
#   3. Ask KEY_METRICS_PROMPT to extract structured data
#
# Returns a list of metric dicts ready to render as cards in the UI.
# ---------------------------------------------------------------------------
def extract_key_metrics(
    index_manager,
    doc_id: str,
    tree: dict,
) -> list[dict]:
    """
    Auto-extract Revenue, PAT, EPS, and D/E Ratio from the document.

    Uses a targeted tree navigation query followed by the KEY_METRICS_PROMPT
    to extract structured data. Each metric includes a citation so the user
    can verify the source page.

    Parameters
    ----------
    index_manager : IndexManager
        The IndexManager instance.
    doc_id : str
        The doc_id for the current document.
    tree : dict
        The full PageIndex tree.

    Returns
    -------
    list[dict]
        List of up to 4 metric dicts:
        [
          {
            "metric":     "Revenue from Operations",
            "value":      "₹9,14,855 Cr",
            "section":    "Financial Highlights",
            "page_range": "33-35"
          },
          ...
        ]
        Metrics not found will have "value": null.
    """

    # Use a specific query to navigate to financial summary sections.
    # Annual reports almost always have a dedicated "Financial Highlights"
    # or "Key Financial Data" section near the front.
    metrics_query = (
        "financial highlights summary revenue income profit after tax PAT "
        "EPS earnings per share debt equity ratio key financial data"
    )
    sections = navigate_tree(tree, metrics_query)

    # Fetch the relevant pages
    context_blocks = []
    for section in sections:
        start = section.get("start_page", 1)
        end = section.get("end_page", start)
        page_range = f"{start}-{end}" if start != end else str(start)
        page_text = index_manager.get_page_content(doc_id, page_range)
        context_blocks.append(page_text)

    full_context = "\n\n".join(context_blocks)

    # Ask the LLM to extract structured metric data from the fetched text
    messages = [
        {
            "role": "user",
            "content": (
                f"{KEY_METRICS_PROMPT}\n\n"
                f"Document text:\n{full_context}"
            ),
        }
    ]

    logger.info("Extracting key metrics...")
    raw = _call_llm(messages)

    # Strip potential markdown fences before parsing
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        metrics = json.loads(cleaned)
        logger.info("Extracted %d metric(s)", len(metrics))
        return metrics
    except json.JSONDecodeError:
        logger.warning("Key metrics JSON parse failed. Returning empty list.")
        return []
