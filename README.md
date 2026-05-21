# ReportReader — Vectorless RAG for Indian Annual Reports

Ask plain-English questions about any Indian company's annual report (PDF) and get answers with exact section and page citations. No vector database. No embeddings. No chunking. Just an LLM reasoning over the document's natural structure using **PageIndex**.

---

## What is Vectorless RAG?

Traditional RAG (Retrieval-Augmented Generation) works by:
1. Splitting a document into fixed-size chunks
2. Embedding each chunk into a vector
3. At query time, embedding the question and finding the closest chunks by cosine similarity
4. Feeding those chunks to an LLM to generate an answer

This works reasonably well for short documents — but it has fundamental problems with long, structured documents like annual reports.

**Vectorless RAG** throws out the embedding pipeline entirely. Instead of searching by similarity, it uses the **LLM itself to reason about where to look** in the document.

---

## How PageIndex Works

PageIndex builds a **hierarchical tree index** from the document — essentially a smart table of contents where every node has a title, a page range, and an LLM-generated summary.

```
PDF
 └── Annual Report FY24
      ├── Board's Report                     [p.1–15]
      │    ├── Financial Performance         [p.3–6]
      │    └── Dividend & Reserves           [p.7–8]
      ├── Management Discussion & Analysis   [p.16–40]
      │    ├── Industry Overview             [p.16–20]
      │    ├── Segment Performance           [p.21–30]
      │    └── Risks & Concerns             [p.31–35]
      └── Financial Statements              [p.41–200]
           ├── Standalone P&L               [p.41–45]
           └── Consolidated Balance Sheet   [p.46–52]
```

When a question comes in, the LLM **navigates this tree step by step** — like a human expert flipping to the right chapter:

```
User: "What is the Debt-to-Equity ratio?"

LLM looks at root node → decides to go into Financial Statements
  → looks at children → picks Standalone P&L
    → retrieves p.41–45 → generates answer with citation
```

### The full pipeline

```
PDF upload
    ↓
Parse PDF with PyMuPDF
    ↓
pageindex.page_index_main(pdf_path)
    → LLM reads sections and builds tree of { title, start_page, end_page, summary, children }
    ↓
User asks a question
    ↓
LLM reasons over tree → navigates to the right node(s)
    → retrieves raw text from those page ranges
    ↓
LLM generates answer with citations: [Section Name, p.X–Y]
```

**No embeddings. No Pinecone. No chunking. No cosine similarity.**

---

## Vector Search vs PageIndex — When to Use Each

| | Vector Search (Traditional RAG) | PageIndex (Vectorless RAG) |
|---|---|---|
| **How it finds content** | Cosine similarity between embeddings | LLM reasons over a document tree |
| **Setup** | Needs embedding model + vector DB (Pinecone, Weaviate, Chroma…) | Just an LLM + the document |
| **Chunking** | Splits document into fixed-size chunks | Preserves document's natural structure |
| **Citations** | Hard — you know what text matched, not where it lives | Exact — every answer traces to a node → page range |
| **Multi-section reasoning** | Weak — each chunk is retrieved independently | Strong — LLM can navigate across sections |
| **Speed at query time** | Very fast (ANN search) | Slower (LLM reasoning steps) |
| **Cost** | Low per query after indexing | Higher per query (more LLM calls) |
| **Best for** | Large corpora, many short documents, semantic search | Single long structured document, deep Q&A |

### ✅ Use PageIndex when

- The document is **long and structured** — research papers, legal contracts, annual reports, RFPs, technical manuals
- Questions require **multi-step reasoning** across sections (e.g. "How does the revenue growth in segment A compare to the risk factors mentioned in the MD&A?")
- You need **explainable, traceable retrieval** — the user or auditor needs to verify the source
- You want to **avoid infrastructure** — no vector DB to spin up, no embeddings to maintain

### ✅ Use Vector Search when

- You have a **large corpus of many documents** and need to find which documents are relevant
- Questions are **factual and localized** — the answer lives in one place
- **Latency matters** — sub-second query responses are required
- You need **fuzzy/semantic matching** across varied phrasings

---

## Why This Matters for Indian Annual Reports

Indian annual reports (BSE/NSE filings, DRHPs, investor presentations) are 200–400 page PDFs with a well-defined structure: Board's Report, MD&A, Corporate Governance, Financial Statements, Notes to Accounts. PageIndex exploits this structure directly.

Traditional RAG chunks these arbitrarily — a chunk might start mid-sentence in the middle of a table, losing the row headers. PageIndex navigates to "Notes to Accounts, p.142" and reads the whole note.

---

## Tech Stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| PDF parsing + tree index | `pageindex` (pip) + PyMuPDF |
| LLM | Any model via LiteLLM — OpenAI, Anthropic, Ollama, Groq… |
| State | `st.session_state` |

---

## Screens

1. **Upload** — drag-and-drop PDF, progress indicator while tree is built (~60–120s for large reports)
2. **Key Metrics** — auto-extracted Revenue, PAT, EPS, D/E ratio — each with a page citation
3. **Q&A** — free-text questions with suggested chips; answers always cite section + page range

---

## Project Structure

```
vectorless-reportreader/
│
├── app.py            # Streamlit UI — all three screens, sidebar tree map,
│                     # session state management, Q&A chat thread
│
├── indexer.py        # IndexManager class — wraps PageIndexClient
│                     # Builds the tree from a PDF, caches it in workspace/,
│                     # exposes get_tree() and get_page_content()
│
├── retriever.py      # Two-step retrieval pipeline:
│                     #   Step 1 — navigate_tree(): LLM reasons over tree
│                     #            to find relevant section + page range
│                     #   Step 2 — answer_question(): fetches those pages,
│                     #            calls LLM to generate a cited answer
│                     # Also contains extract_key_metrics() for Screen 2
│
├── prompts.py        # All LLM system prompts as named string constants:
│                     #   TREE_NAVIGATION_PROMPT  — which section to look in
│                     #   KEY_METRICS_PROMPT      — extract Revenue/PAT/EPS/D:E
│                     #   QA_SYSTEM_PROMPT        — answer with citations
│
├── requirements.txt  # Python dependencies
├── .env.example      # API key template (copy to .env and fill in)
├── .gitignore        # Excludes .env, workspace/, *.pdf
└── workspace/        # Auto-created — PageIndex caches tree JSON files here
                      # (gitignored — regenerated automatically on first run)
```

### How the files connect

```
app.py
  └── on PDF upload → indexer.py (IndexManager.index_pdf)
                          └── PageIndexClient builds tree → workspace/
  └── on question  → retriever.py (answer_question)
                          ├── navigate_tree() — LLM + prompts.py reads tree
                          ├── IndexManager.get_page_content() — fetch pages
                          └── LLM + prompts.py generates cited answer
  └── on load      → retriever.py (extract_key_metrics)
                          └── same pipeline, targeted at financial metrics
```

---

## How to Run

### 1. Clone and install

```bash
git clone https://github.com/techwithprateek/vectorless-reportreader.git
cd vectorless-reportreader
pip install -r requirements.txt
```

### 2. Configure your LLM

```bash
cp .env.example .env
```

Open `.env` and set your model + API key. Pick one:

```bash
# OpenAI
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-your-key-here

# Anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Ollama (local, free — run `ollama pull llama3.2` first)
LLM_MODEL=ollama/llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

> LiteLLM handles routing — just change `LLM_MODEL` to switch providers.
> Full list of supported models: [docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers)

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### 4. Use it

1. Upload any Indian company annual report PDF (max 50 MB)
2. Wait ~1–2 minutes while PageIndex builds the document tree *(only happens once per document — subsequent loads are instant)*
3. View auto-extracted **Key Metrics** (Revenue, PAT, EPS, D/E Ratio)
4. Switch to **Q&A** and ask anything — every answer cites the section and page

---

*No vector database was harmed in the making of this project.*