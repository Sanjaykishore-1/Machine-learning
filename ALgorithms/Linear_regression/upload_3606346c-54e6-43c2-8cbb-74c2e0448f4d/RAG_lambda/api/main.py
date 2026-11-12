import os
import re
from typing import List, Optional

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from rank_bm25 import BM25Okapi

app = FastAPI(title="Customer Document RAG API", version="1.0.0")

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
chroma = chromadb.HttpClient(
    host=os.environ["CHROMA_HOST"],
    port=int(os.getenv("CHROMA_PORT", "8000")),
    ssl=os.getenv("CHROMA_SSL", "false").lower() == "true",
)
collection = chroma.get_or_create_collection(
    name=os.getenv("CHROMA_COLLECTION", "customer_documents"),
    metadata={"hnsw:space": "cosine"},
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "8"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "8"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_id: Optional[str] = None


class Source(BaseModel):
    document_id: str
    page: int
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9_/-]+", text.lower())


def reciprocal_rank_fusion(vector_ids, lexical_ids, k=60):
    scores = {}
    for rank, chunk_id in enumerate(vector_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, chunk_id in enumerate(lexical_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return [item[0] for item in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def retrieve_hybrid(question: str, document_id: Optional[str] = None):
    where = {"document_id": document_id} if document_id else None

    query_embedding = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    ).data[0].embedding

    vector_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=VECTOR_TOP_K,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    vector_ids = vector_result["ids"][0]
    vector_docs = vector_result["documents"][0]
    vector_metas = vector_result["metadatas"][0]
    vector_distances = vector_result["distances"][0]

    # Demo hybrid implementation: BM25 is computed over the selected document
    # corpus fetched from Chroma. For large corpora, use OpenSearch BM25 or
    # another dedicated lexical index instead of loading all chunks per request.
    corpus = collection.get(
        where=where,
        include=["documents", "metadatas"],
    )
    corpus_ids = corpus["ids"]
    corpus_docs = corpus["documents"] or []
    corpus_metas = corpus["metadatas"] or []

    lexical_ids = []
    if corpus_docs:
        tokenized_corpus = [tokenize(doc or "") for doc in corpus_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = tokenize(question)
        if query_tokens:
            scores = bm25.get_scores(query_tokens)
            ranked_indices = sorted(
                range(len(scores)), key=lambda idx: scores[idx], reverse=True
            )[:BM25_TOP_K]
            lexical_ids = [corpus_ids[idx] for idx in ranked_indices if scores[idx] > 0]

    fused_ids = reciprocal_rank_fusion(vector_ids, lexical_ids)[:FINAL_TOP_K]
    lookup = {
        chunk_id: {"text": doc, "metadata": meta}
        for chunk_id, doc, meta in zip(corpus_ids, corpus_docs, corpus_metas)
    }
    # Include vector candidates even if corpus response is unexpectedly partial.
    for chunk_id, doc, meta in zip(vector_ids, vector_docs, vector_metas):
        lookup.setdefault(chunk_id, {"text": doc, "metadata": meta})

    selected = []
    for chunk_id in fused_ids:
        item = lookup.get(chunk_id)
        if item:
            selected.append((chunk_id, item["text"], item["metadata"]))

    # Similarity threshold is checked on the best vector result.
    # Chroma cosine distance: lower is closer; 1 - distance is a rough similarity.
    best_similarity = 1.0 - vector_distances[0] if vector_distances else 0.0
    if not selected or best_similarity < MIN_SIMILARITY:
        return []

    return selected


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        retrieved = retrieve_hybrid(request.question, request.document_id)
        if not retrieved:
            return AskResponse(
                answer="I couldn't find enough relevant information in the uploaded documents to answer that question.",
                sources=[],
            )

        context_parts = []
        sources = []
        for chunk_id, text, metadata in retrieved:
            page = int(metadata.get("page", 0))
            doc_id = metadata.get("document_id", "unknown")
            context_parts.append(
                f"[Source: {doc_id}, page {page}]\n{text}"
            )
            sources.append(Source(
                document_id=doc_id,
                page=page,
                chunk_id=chunk_id,
            ))

        context = "\n\n---\n\n".join(context_parts)
        prompt = f"""
You answer questions using only the supplied document context.
If the context does not contain the answer, say that the documents do not
provide enough information. Do not use outside knowledge to fill gaps.
Cite supporting pages in the answer using [page N].

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{request.question}

ANSWER:
""".strip()

        completion = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful document question-answering assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        answer = completion.choices[0].message.content or "I couldn't generate an answer."
        return AskResponse(answer=answer, sources=sources)

    except Exception as exc:
        # Log details server-side; do not expose credentials or internals to clients.
        print(f"RAG request failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="The document question service is temporarily unavailable.")
