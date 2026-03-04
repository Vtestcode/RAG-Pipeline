# RAG Sample Chatbot

This project includes a sample chatbot built on top of a RAG pipeline.

## 1) Configure environment

In `key.env`, set at least:

- `OPENAI_API_KEY=your_key_here`
- `RAG_PDF_PATH=C:\\path\\to\\your\\document.pdf`

If `RAG_PDF_PATH` is not set, code falls back to:

- `s3://rag-pipeline-vishal/Vishal.pdf`

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Run sample chatbot (Streamlit)

```bash
streamlit run app/chatbot.py
```

## 4) Optional API mode (FastAPI)

```bash
uvicorn main:app --reload
```

POST query example:

- Endpoint: `POST /ask?query=your question`
