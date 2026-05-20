# ReportReader
**Product spec · demo level · v1.0**

---

## What it does

ReportReader lets anyone upload an Indian company's annual report (PDF) and instantly ask plain-English questions — with every answer citing the exact section and page it came from. Powered by PageIndex: no vector database, no chunking, just reasoning over the document's natural structure.

---

## The problem

Indian annual reports (BSE/NSE filings, DRHP, investor presentations) are 200–400 page PDFs. Finding a specific number, management comment, or risk disclosure means scrolling endlessly. Traditional RAG chunks documents arbitrarily and retrieves by similarity — which misses context and loses page references. ReportReader uses PageIndex to reason over the document like a human expert would.

---

## Who it's for (demo)

Retail investors, CA students, journalists, research analysts — anyone who needs to extract insight from a dense financial PDF fast.

---

## What's in scope

| In | Out |
|---|---|
| Single PDF upload | Multi-doc comparison |
| PageIndex tree index (built on upload) | Vector DB or embeddings |
| Q&A with section + page citations | User accounts |
| Auto key metrics extraction | Charts / visualizations |
| Streamlit UI | Mobile app |
| Indian company context (₹, SEBI, BSE/NSE) | Real-time stock data |

---

## How PageIndex works here

PageIndex is **vectorless, reasoning-based RAG**. Instead of chunking text and doing similarity search, it:

1. Reads the PDF and builds a **hierarchical tree index** — essentially a smart table of contents with summaries at each node
2. When a question comes in, an LLM **reasons over the tree** to find the right node(s)
3. Retrieves only the relevant sections with their **page and section references**

This is why citations are accurate: the retrieval knows *where in the document* it looked, not just what text matched a vector.

```
PDF upload
    ↓
pageindex.page_index_main(pdf_path, opt)
    → builds tree: { title, start_index, end_index, summary, nodes: [...] }
    ↓
User asks a question
    ↓
LLM reasons over tree → identifies relevant node(s)
    → retrieves text from start_index to end_index pages
    ↓
Answer generated with citations: section title + page range
```

The tree node structure looks like:
```json
{
  "title": "Financial Highlights",
  "node_id": "0003",
  "start_index": 18,
  "end_index": 24,
  "summary": "Key financial metrics for FY2024...",
  "nodes": [...]
}
```

Every answer traces back to a node → which has a page range → which is shown to the user.

---

## Three screens

### Screen 1 — Upload
- Drag-and-drop or file picker for PDF (max 50 MB)
- "Processing…" spinner while PageIndex builds the tree (this takes 30–90 seconds for a large report — set expectations clearly)
- On completion: show company name + fiscal year auto-detected from the tree's doc description
- Optional: one pre-loaded sample report (e.g. Reliance Industries FY24) for instant demo

### Screen 2 — Key Metrics
- Auto-extracted on load using a targeted question to the PageIndex RAG pipeline
- Displayed as 4 metric cards, each with value + citation:
  - Total Income / Revenue from Operations
  - Profit After Tax (PAT)
  - Basic EPS
  - Debt-to-Equity Ratio
- Each card shows: `₹9,14,855 Cr` and `→ p.33, Financial Highlights`

### Screen 3 — Q&A
- Text input: "Ask anything about this report…"
- 4–5 suggested question chips to remove the blank-page problem:
  - "What are the key risk factors?"
  - "What did management say about future outlook?"
  - "What is the dividend declared?"
  - "How did revenue change vs last year?"
- Answer displayed with inline citations: **section name + page range**
- Full conversation history preserved in `st.session_state` across turns

---

## Key metrics extraction prompt

```
You are analyzing an Indian company annual report.
Extract these values if present:
- Total Income or Revenue from Operations (in ₹ Crore or ₹ Lakh)
- Profit After Tax (PAT)
- Basic EPS
- Debt-to-Equity Ratio

For each, return the value and the section/page where you found it.
Format: JSON array with { "metric", "value", "section", "page_range" }.
If not found, return null for that metric. Do not guess.
```

---

## Q&A system prompt

```
You are a financial analyst assistant for Indian company annual reports.

Rules:
1. Answer only from the retrieved document sections provided.
2. Every factual claim must cite its source:
   → format: [Section Name, p.X–Y]
3. Use Indian financial terminology: ₹ Crore, PAT, EBITDA,
   promoter holding, SEBI, BSE/NSE, standalone vs consolidated, etc.
4. Keep answers to 3–5 sentences unless detail is asked for.
5. If the answer is not in the retrieved sections, say so clearly.
   Do not hallucinate figures.
```

---

## Tech stack

| Layer | Choice | Reason |
|---|---|---|
| UI | Streamlit | Fastest to demo, no frontend code needed |
| PDF → tree index | `pageindex` (pip) + PyMuPDF | PageIndex uses PyMuPDF internally for PDF parsing |
| LLM for indexing | Any via LiteLLM — use `gpt-4o` or `claude-3-5-sonnet` | PageIndex supports multi-LLM via LiteLLM |
| LLM for Q&A | Same model | Consistency; one API key |
| State | `st.session_state` | Persist tree + chat history across turns |

---

## File structure

```
reportreader/
├── app.py            # Streamlit app — all UI logic
├── indexer.py        # Wraps pageindex: build tree from PDF
├── retriever.py      # Wraps pageindex: tree search + answer generation
├── prompts.py        # System prompts as constants
├── requirements.txt
├── .env              # API keys
└── sample/
    └── reliance_fy24.pdf
```

---

## requirements.txt

```
streamlit
pageindex          # pip install pageindex
pymupdf
openai             # or anthropic — pageindex uses litellm
python-dotenv
```

---

## .env

```
OPENAI_API_KEY=your_key_here
# or
ANTHROPIC_API_KEY=your_key_here
```

---

## UX note: indexing time

Building the PageIndex tree is not instant — it makes multiple LLM calls to summarize sections. For a 300-page report, expect 60–120 seconds. Show a progress message like:

> "Building document index… this takes about a minute for large reports."

Cache the tree in `st.session_state` so Q&A turns are fast after that.

---

## Demo script (2 minutes)

1. Open app → click "Try sample report" (Reliance FY24 loads, tree already cached)
2. Key metrics appear: Revenue ₹9,14,855 Cr `→ Financial Highlights, p.33`
3. Click suggested chip: "What are the key risk factors?"
4. Answer cites: `[Risk Factors, p.47–51]`
5. Type: "What did management say about the telecom business?"
6. Answer cites MD&A section with page range
7. Point out: **"PageIndex read the document like a human — it navigated the structure, not a vector index. Every answer is traceable."**

---

## What makes this a good PageIndex demo

- Indian annual reports are long enough (200–400 pages) that chunking-based RAG struggles — PageIndex's tree navigation is visibly better
- Section + page citations are shown explicitly — the audience can open the PDF and verify
- No vector DB setup required — easier to run locally and explain to a non-technical audience
- The tree structure itself can be shown in the sidebar as a "document map" to make the indexing step tangible

---

*Build time estimate: 1 focused day for a working demo.*
