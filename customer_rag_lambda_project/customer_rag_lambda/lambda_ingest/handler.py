import json
import logging
import os
import re
from typing import List, Dict, Any

import boto3
import fitz
import chromadb
from openai import OpenAI

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_HOST = os.environ["CHROMA_HOST"]
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_SSL = os.getenv("CHROMA_SSL", "false").lower() == "true"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "customer_documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

chroma = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    ssl=CHROMA_SSL,
)
collection = chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def extract_pdf_text(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract page text while preserving page numbers (1-based)."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": page_number, "text": text})
    return pages


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split each page into overlapping character chunks; retain page metadata."""
    chunks = []
    for page in pages:
        text = re.sub(r"\s+", " ", page["text"]).strip()
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "page": page["page"],
                    "text": chunk_text,
                })
            if end >= len(text):
                break
            start = max(0, end - CHUNK_OVERLAP)
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = []
    for offset in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[offset:offset + EMBED_BATCH_SIZE]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def lambda_handler(event, context):
    """
    Trigger: S3 ObjectCreated event for PDFs.
    Configure this Lambda with S3 read permission and invoke permission.
    """
    processed = 0
    for record in event.get("Records", []):
        if record.get("eventSource") != "aws:s3":
            continue

        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        from urllib.parse import unquote_plus
        key = unquote_plus(key)

        if not key.lower().endswith(".pdf"):
            logger.info("Skipping non-PDF object: s3://%s/%s", bucket, key)
            continue

        # Stable document ID makes retries idempotent.
        document_id = f"{bucket}/{key}"
        logger.info("Processing s3://%s/%s", bucket, key)

        obj = s3.get_object(Bucket=bucket, Key=key)
        pdf_bytes = obj["Body"].read()

        pages = extract_pdf_text(pdf_bytes)
        chunks = chunk_pages(pages)
        if not chunks:
            logger.warning("No extractable text in %s", document_id)
            continue

        # Stable chunk IDs + upsert prevent duplicate records on S3/Lambda retries.
        ids = [
            f"{document_id}::page-{chunk['page']}::chunk-{i}"
            for i, chunk in enumerate(chunks)
        ]
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts(texts)
        metadatas = [
            {
                "document_id": document_id,
                "s3_bucket": bucket,
                "s3_key": key,
                "page": int(chunk["page"]),
                "chunk_index": int(i),
            }
            for i, chunk in enumerate(chunks)
        ]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        processed += 1
        logger.info("Stored %d chunks for %s", len(chunks), document_id)

    return {
        "statusCode": 200,
        "body": json.dumps({"processed_documents": processed}),
    }
