# Customer Document RAG — S3 + Lambda + FastAPI + ChromaDB

This is a practical starter implementation of a document-question-answering RAG system.

## Architecture

### Ingestion
1. React uploads a PDF to FastAPI (or uses a presigned S3 URL).
2. FastAPI stores the PDF in S3.
3. S3 `ObjectCreated` event invokes the Lambda container.
4. Lambda extracts PDF text, chunks it, creates embeddings, and upserts chunks into a persistent ChromaDB server.

### Question answering
1. React sends a question to `POST /ask`.
2. FastAPI performs vector retrieval and BM25 keyword retrieval.
3. Reciprocal Rank Fusion combines the two result lists.
4. The selected context is sent to the LLM.
5. FastAPI returns the answer and source page references.

## Important deployment decisions

- **Do not run a persistent ChromaDB database inside Lambda.** Lambda storage is temporary. Run ChromaDB as a separately hosted service with persistent storage, or use a managed vector database.
- The included BM25 implementation fetches the corpus from ChromaDB for each query. This is suitable only for a small learning/demo corpus. For production-scale hybrid search, use a dedicated lexical index such as OpenSearch alongside ChromaDB, then fuse the ranked results.
- The sample uses OpenAI embeddings and chat completion. Keep API keys in AWS Secrets Manager or secure Lambda/ECS environment configuration; never commit secrets.
- Configure S3 event notifications with a prefix such as `uploads/` and suffix `.pdf`. Avoid writing processed files back into the same triggering prefix, or you may create recursive invocations.
- S3 notifications and Lambda processing can be delivered more than once. Stable chunk IDs and Chroma `upsert` make retries idempotent for the same document version. For production, include the S3 object VersionId or ETag in the document identity and remove stale chunks when a PDF is replaced.
- Add SQS between S3 and Lambda for burst handling, retries, and a dead-letter queue when ingestion volume grows.
- Scanned PDFs need OCR (for example, Amazon Textract); PyMuPDF text extraction alone will not read text embedded only in images.
- Large PDFs may exceed Lambda time or memory limits. Use Lambda container images and suitable memory/timeout settings; move long-running jobs to ECS/Fargate or Step Functions when necessary.

## 1. Run the API locally

Start a persistent Chroma server (for local testing):

```bash
docker run -d --name chroma \
  -p 8000:8000 \
  -v chroma_data:/chroma/chroma \
  chromadb/chroma:0.6.3
```

Create a Python environment and install API dependencies:

```bash
cd api
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables from the project root (PowerShell example):

```powershell
$env:OPENAI_API_KEY="your-key"
$env:CHROMA_HOST="localhost"
$env:CHROMA_PORT="8000"
$env:CHROMA_SSL="false"
uvicorn main:app --reload
```

API docs: `http://127.0.0.1:8000/docs`

## 2. Build and deploy the Lambda container

From `lambda_ingest/`:

```bash
docker build -t customer-rag-ingest .
```

Push this image to Amazon ECR, then create a Lambda function from the ECR image. Configure:
- Runtime: container image (Python 3.12 base image)
- Handler is already set in the Dockerfile
- Memory: begin with 2048 MB and benchmark
- Timeout: set an appropriate value up to Lambda's 15-minute maximum
- Environment variables: `OPENAI_API_KEY`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SSL`, `CHROMA_COLLECTION`
- IAM role: `s3:GetObject` on the upload bucket/prefix and CloudWatch Logs permissions
- Network: Lambda must be able to reach the ChromaDB endpoint and the embedding API. If ChromaDB is in a private VPC, configure Lambda VPC access and networking accordingly.

Grant S3 permission to invoke the Lambda and configure an `ObjectCreated` notification for `uploads/*.pdf`.

## 3. Example upload flow

Your FastAPI upload endpoint should save the file to S3. A minimal endpoint can be added using `boto3`:

```python
import boto3
from fastapi import UploadFile, File, HTTPException

s3 = boto3.client("s3")
UPLOAD_BUCKET = os.environ["UPLOAD_BUCKET"]

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    key = f"uploads/{uuid.uuid4()}-{file.filename}"
    s3.upload_fileobj(file.file, UPLOAD_BUCKET, key)
    return {"message": "Upload accepted", "s3_key": key}
```

Add `import uuid` and `import os`, configure `UPLOAD_BUCKET`, and grant the API task role `s3:PutObject` on the upload prefix. In a production frontend, consider presigned S3 uploads for large files.

## 4. Test a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the payment status of invoice INV-1025?"}'
```

Optional: restrict the question to one uploaded document:

```json
{
  "question": "What is the payment status?",
  "document_id": "your-bucket/uploads/example.pdf"
}
```

## 5. Production checklist

- Authentication and authorization; users must only retrieve documents they are allowed to access.
- Tenant/user IDs in metadata and mandatory filters on every retrieval.
- File size limits, PDF validation, malware scanning, and upload quotas.
- SQS queue, retry policy, and dead-letter queue for ingestion.
- S3 versioning and lifecycle policy.
- Persistent, backed-up vector storage.
- OCR for scanned PDFs.
- Evaluation dataset for Recall@K, Precision@K, answer faithfulness, and latency.
- Secrets Manager, TLS, private networking, and CloudWatch alarms.
- Document deletion endpoint that removes all associated vector chunks.
