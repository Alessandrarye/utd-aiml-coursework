# Merrick Bank Customer Retention Agent

RAG-based retention assistant built with LangChain, FAISS, Hugging Face embeddings, and Gradio for the UTD AI/ML Bootcamp agentic AI capstone session.

It combines structured churn data (10,000 customers plus sentiment labels) with the bank's policy documents to answer retention questions with citations, personalized to a customer ID.

## Layout

```
project-merrick-retention-agent/
├── Merrick_Retention_Agent.ipynb   # the assignment notebook, Tasks 1 to 6
├── app.py                          # same code as a standalone Gradio app
├── requirements.txt
├── .env.example                    # copy to .env and add OPENAI_API_KEY
├── data/                           # Churn_Modeling1.csv, Churn_With_Sentiment.csv (gitignored)
├── docs/                           # FAQ, policies, product summary
└── index/                          # FAISS index, generated on first run (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit .env and paste in your OpenAI key
```

To use Ollama instead of OpenAI, skip the key, run `ollama pull llama3.2`, and set `LLM_PROVIDER=ollama` in `.env`.

The first run downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB) from Hugging Face.

## Run

- **Notebook:** open `Merrick_Retention_Agent.ipynb` and Run All.
- **App:** `python app.py`, then open http://127.0.0.1:7860. Set `REBUILD_INDEX=1` to re-embed after changing anything in `docs/`.

## How it works

| Task | Where |
|---|---|
| 1. Data preparation | `prepare_folders`, `verify_documents` |
| 2. Ingestion and embedding | `load_documents`, `split_documents`, `get_embeddings`, `build_or_load_index` |
| 3. Customer data | `CustomerDB.get_context(customer_id)` |
| 4. RAG chain | `build_chain` (Runnable composition), `SYSTEM_PROMPT`, `get_llm` |
| 5. Gradio UI | `build_ui` |
| 6. Evaluation | `run_evaluation`, `automated_checks`, `judge_answer` |

Sample queries to try:

- How can I waive my annual fee?
- What should we offer to retain customer 15634602?
- Explain the difference between Gold and Platinum cards.
