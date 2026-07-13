# 🚗🏍️ Bike & Car Price/Mileage RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot: ask about a bike or
car's **on-road price**, **ex-showroom price**, and **mileage**, and get answers
grounded in your own PDF data — not hallucinated.

## Stack

| Piece         | Tool                                   | Account needed? |
|---------------|-----------------------------------------|------------------|
| Vector DB     | **ChromaDB** (local, persistent client) | ❌ No — runs on disk |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2` (local) | ❌ No |
| LLM (answers) | **OpenRouter** (OpenAI-compatible API)  | ✅ Yes — free key at https://openrouter.ai/keys |
| Frontend      | **Streamlit**                           | ❌ No |

> Why this split? You said you don't have a ChromaDB (cloud) account and want
> to use OpenRouter for the API key. ChromaDB's `PersistentClient` needs **no**
> account — it just writes to a local folder — so that's the natural fit.
> OpenRouter is used only for the LLM that generates the final answer.

## Folder structure

```
project/
├── data/
│   ├── generate_dataset.py   # sample bike/car data (Python list of dicts)
│   ├── build_pdfs.py         # builds the PDFs below from the dataset
│   ├── bikes_data.pdf        # ready-to-use sample data
│   └── cars_data.pdf         # ready-to-use sample data
├── notebook/
│   └── RAG_Bike_Car_Chatbot_Practice.ipynb   # step-by-step learning notebook
├── backend/
│   ├── config.py              # env-var driven settings
│   ├── ingest.py              # PDF -> chunks -> embeddings -> ChromaDB
│   ├── rag_pipeline.py        # retrieval + OpenRouter generation (RAGPipeline class)
│   └── requirements.txt
├── frontend/
│   └── app.py                 # Streamlit chat UI
├── .env.example                # copy to .env and add your OpenRouter key
└── README.md
```

## Quick start

```bash
# 1. Create & activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\\Scripts\\activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Add your OpenRouter API key
cp .env.example .env
# then edit .env and paste your key from https://openrouter.ai/keys

# 4. Build the vector database from the sample PDFs (run once, or whenever data changes)
cd backend
python ingest.py

# 5. Launch the chatbot UI
cd ../frontend
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`) and ask
things like:
- "What is the on-road price and mileage of Honda Activa 6G?"
- "Compare Tata Nexon and Hyundai Creta"
- "Cheapest electric bike in the dataset?"

## Using your own data

Replace `data/bikes_data.pdf` / `data/cars_data.pdf` with your own PDFs (or add
more PDFs to the `data/` folder), then re-run `python backend/ingest.py` to
rebuild the vector store. For best retrieval quality, keep the same
"one-vehicle-per-paragraph, clearly labelled fields" structure used in the
sample PDFs — see `data/build_pdfs.py` for the exact format.

## Swapping the LLM model

Edit `OPENROUTER_MODEL` in `.env` (or `backend/config.py`). Browse available
models — including several free ones — at https://openrouter.ai/models.

## Notes / next steps for production

- The sample price data is **illustrative**, not live — verify before real use.
- `ingest.py` rebuilds the entire collection each run (simple & safe for a
  small dataset). For large/frequently-updated datasets, switch to incremental
  upserts (`collection.upsert(...)`) keyed by a stable vehicle ID.
- Add authentication / rate limiting before deploying the Streamlit app publicly.
- Consider caching frequent queries and logging retrieved-chunk relevance to
  monitor RAG quality over time.
