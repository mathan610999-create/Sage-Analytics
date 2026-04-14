# 🌿 Sage — Wise advice from your data

Sage is an AI-powered decision helper that translates business data into plain English insights and options — so anyone can understand their data and make better decisions, without needing to be an analyst.

**Not a decision maker. A decision helper.**

---

## What makes Sage different

Most analytics tools are built for analysts. Sage is built for the person who receives the report — the business owner, the manager, the executive who needs to act on data but doesn't have time to learn SQL or build dashboards.

Sage doesn't tell you what to do. It tells you what the data says and gives you options to consider. The decision is always yours.

---

## What Sage does

- Upload any CSV or Excel file
- Auto-analyses the data — EDA, KPIs, patterns, anomalies
- Generates a plain English business briefing
- Answers follow-up questions in natural language
- Remembers context across the conversation
- Generates charts automatically when relevant

---

## Tech stack

- **Python** — core language
- **LangGraph** — ReAct agent loop and memory
- **LangChain** — tool orchestration
- **Claude API** (Anthropic) — plain English reasoning
- **Pandas** — data cleaning and EDA
- **Plotly** — interactive charts
- **ChromaDB** — vector database for RAG (Phase 3)
- **Streamlit** — chat UI and deployment

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# Edit .env with your Anthropic API key

# 3. Generate sample data
python generate_data.py

# 4. Run the app
streamlit run app.py
```

---

## Project phases

- [x] **Phase 1** — File upload + EDA + plain English briefing + chat
- [ ] **Phase 2** — NL-to-SQL + chart generation on demand
- [ ] **Phase 3** — RAG — upload PDF documents for context-aware answers
- [ ] **Phase 4** — ML models — forecasting and clustering
- [ ] **Phase 5** — Deploy to Streamlit Cloud

---

## The philosophy

> "I don't tell you what to decide. I make sure you have everything you need to decide well."
