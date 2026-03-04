import tempfile
from pathlib import Path
from urllib.parse import urlparse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


def _download_s3_pdf(s3_uri: str) -> str:
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "boto3 is required for S3 paths. Install it with: py -m pip install boto3"
        ) from exc

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(
            f"Invalid S3 URI '{s3_uri}'. Expected format: s3://bucket/path/file.pdf"
        )

    suffix = Path(key).suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    boto3.client("s3").download_file(bucket, key, tmp_path)
    return tmp_path


def _resolve_pdf_path(file_path: str) -> str:
    if file_path.startswith("s3://"):
        return _download_s3_pdf(file_path)

    local_path = Path(file_path)
    if not local_path.exists():
        raise FileNotFoundError(
            f"RAG PDF not found at '{local_path}'. Set RAG_PDF_PATH to a valid local path or S3 URI."
        )
    return str(local_path)


def build_vector_store(file_path):
    """Load a PDF, split it into chunks, embed chunks, and return a FAISS vector store."""
    resolved_path = _resolve_pdf_path(file_path)
    loader = PyPDFLoader(resolved_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store
