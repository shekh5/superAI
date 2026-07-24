# Session Document Retrieval

> This page also describes the backward-compatible Redis lexical mode. When
> `EVIDENCE_WORKSPACE_ENABLED=true`, originals, ownership, queued ingestion, hybrid retrieval,
> and citation previews use the production architecture documented in
> [evidence-workspace.md](evidence-workspace.md).

## Scope

SuperAI accepts several text-bearing document formats, extracts their content once, and persists
searchable chunks in Redis. Later questions in the same session retrieve only relevant passages
and send those passages to Gemini as high-priority, untrusted document context. Complete documents
are never inserted into every prompt.

Supported formats and citations:

| Format | Parser | Citation locator |
|---|---|---|
| PDF (`.pdf`) | `pypdf` | page number |
| Word (`.docx`) | `python-docx` | block range |
| Text/Markdown (`.txt`, `.md`) | UTF-8 decoder | section |
| CSV/TSV (`.csv`, `.tsv`) | Python CSV parser | row range |
| Excel (`.xlsx`) | `openpyxl` read-only mode | sheet and row range |
| PowerPoint (`.pptx`) | `python-pptx` | slide number |

Legacy binary Office formats (`.doc`, `.xls`, `.ppt`) are rejected. Convert them to their modern
OOXML equivalents before uploading.

This version intentionally uses local lexical ranking instead of an embedding API. It avoids
another paid quota and works with the existing FastAPI plus Redis deployment. Semantic and
cross-language retrieval can later be added behind the same storage interface with embeddings and
a vector index.

## Processing flow

```text
Document upload
  -> validate extension, byte size, container, unit count, and extracted-text size
  -> run the format-specific text extractor
  -> preserve page, block, row, sheet, section, or slide location
  -> create overlapping chunks
  -> store Redis metadata and chunk lists with TTL
  -> attach document ID to the chat session

Later question
  -> load documents owned by the session
  -> rank chunks by query terms, document frequency, and filename
  -> use representative chunks for generic explanation requests
  -> inject bounded document_context before the current goal
  -> require an exact [filename, locator] citation in a satisfied answer
```

Passages are XML-escaped and marked `trust="untrusted"`. Instructions found inside documents
cannot extend the runtime tool registry, which has no filesystem, shell, Git, or code-editing tool.

## API

Upload any supported file:

```bash
curl -X POST \
  -F "file=@report.docx" \
  http://localhost:8000/chain/session/my-session/documents
```

List the session's documents:

```bash
curl http://localhost:8000/chain/session/my-session/documents
```

Ask a question later using the same session ID:

```bash
curl -X POST \
  "http://localhost:8000/chain/run?session_id=my-session&goal=How+much+did+revenue+grow%3F"
```

Delete extracted content:

```bash
curl -X DELETE \
  http://localhost:8000/chain/session/my-session/documents/DOCUMENT_ID
```

The browser chat exposes the same flow through its `Files` button. Its badge shows how many
documents are attached to the selected session.

## Redis records

```text
session:{session_id}:documents       -> ordered list of document IDs
document:{document_id}:metadata      -> type, filename, unit/chunk counts, session, timestamps
document:{document_id}:chunks        -> extracted locator-preserving chunks
```

All three records use `DOCUMENT_TTL_SECONDS`, which defaults to 30 days. The existing append-only
Redis volume preserves them through normal container recreation.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `DOCUMENT_MAX_FILE_BYTES` | `10485760` | Maximum upload read (10 MB) |
| `DOCUMENT_MAX_UNITS` | `200` | Page/sheet/slide limit; text row/block limit is 50x |
| `DOCUMENT_MAX_EXTRACTED_CHARS` | `2000000` | Extraction expansion guard |
| `DOCUMENT_CHUNK_WORDS` | `450` | Words per chunk |
| `DOCUMENT_CHUNK_OVERLAP_WORDS` | `75` | Adjacent chunk overlap |
| `DOCUMENT_RETRIEVAL_CHUNKS` | `8` | Maximum passages retrieved per question |
| `DOCUMENT_RETRIEVAL_MAX_CHARS` | `14000` | Document-context character cap |
| `DOCUMENT_MAX_PER_SESSION` | `10` | Upload count guard |
| `DOCUMENT_TTL_SECONDS` | `2592000` | Metadata and chunk retention |

The former `PDF_*` environment variables remain accepted as compatibility fallbacks. New
deployments should use the `DOCUMENT_*` names.

Set an equivalent request-body limit in the EC2 reverse proxy or load balancer. Application-level
limits run after the multipart parser has started receiving the request.

## Current limitations

- Scanned/image-only content is rejected with an OCR-required message.
- Password-protected PDFs are rejected.
- Text and delimited files must use UTF-8 encoding.
- Retrieval is lexical, so questions should share meaningful terminology with the document.
  Common English and Hindi requests such as "explain this file" and "isko samjhao" use
  representative passages.
- Redis stores extracted text, not original file bytes. Reprocessing or visual preview requires
  another upload.
- A session ID scopes access but is not authentication. Add authenticated user ownership before
  exposing uploads publicly.
- Complex layouts, charts, images, macros, presenter notes, and visual relationships are not
  interpreted. Spreadsheet formulas are extracted as formulas rather than recalculated values.

Production extensions should add malware scanning, OCR, object storage for originals, background
processing, authenticated ownership, hybrid vector plus keyword retrieval, reranking, and per-user
storage quotas.
