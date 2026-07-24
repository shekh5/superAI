# Evidence Workspace

Evidence Workspace is the authenticated production document path. It is additive: with
`EVIDENCE_WORKSPACE_ENABLED=false`, the existing Redis lexical document retriever remains active.

## Architecture

```text
Browser -> FastAPI authentication -> private S3 original
                              \-> Redis/RQ ingestion worker
                                      -> format extraction
                                      -> multilingual-e5-small embeddings
                                      -> Neon PostgreSQL + pgvector

Question -> lexical candidates + vector candidates -> weighted RRF -> bounded document context
         -> Gemini structured answer/claims -> local semantic evidence check -> clickable citation
```

The generative model cannot invent an accessible citation. A claim may reference only a `chunk_id`
included in that request's retrieved context. The server then checks the claim against that exact
passage with the local embedding model. `Supported` means the configured semantic threshold was
met; it is evidence-match confidence, not a guarantee that the source itself is true.

## Required production resources

1. Create a private S3 bucket with Block Public Access and default encryption enabled.
   Configure a 30-day lifecycle expiration as a final safety net; the application also removes
   expired records opportunistically when the owner uses the workspace.
2. Attach an EC2 instance profile that permits `GetObject`, `PutObject`, and `DeleteObject` only for
   that bucket. Do not store AWS access keys in GitHub.
3. Create a Neon PostgreSQL database and enable `vector`:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. Configure HTTPS before setting `COOKIE_SECURE=true`. For a temporary IP-address HTTP demo,
   `COOKIE_SECURE=false` is required, but it must not be treated as production security.
5. Add the GitHub Actions secrets listed in README. The CD job runs `alembic upgrade head` before
   starting an enabled workspace release.

Required runtime values:

| Variable | Purpose |
|---|---|
| `EVIDENCE_WORKSPACE_ENABLED=true` | Select S3/pgvector document path |
| `AUTH_REQUIRED=true` | Require team accounts |
| `DATABASE_URL` | Neon `postgresql+psycopg://...` connection URL |
| `S3_BUCKET` / `AWS_REGION` | Private originals location |
| `BOOTSTRAP_ADMIN_EMAIL` | First administrator, used only when `users` is empty |
| `BOOTSTRAP_ADMIN_PASSWORD` | Temporary first password; user must change it |
| `COOKIE_SECURE` | `true` behind HTTPS |
| `EMBEDDING_MODEL` | Defaults to `intfloat/multilingual-e5-small` |
| `EVIDENCE_SUPPORT_THRESHOLD` | Defaults to `0.70` |

After the first admin logs in and changes their temporary password, create members with:

```http
POST /admin/users
Content-Type: application/json

{"email":"member@example.com","temporary_password":"long-temporary-password","role":"member"}
```

There is no public registration, invitation email, or password-reset flow in this release.
Changing the embedding model requires a 384-dimensional compatible model and re-embedding every
stored chunk; do not change it on an existing index as an ordinary configuration rollout.

## APIs

- `POST /auth/login`, `POST /auth/change-password`, `POST /auth/logout`, `GET /auth/me`
- `POST /admin/users`
- `POST /chain/session/{session_id}/documents` returns `202` and a queued document
- `GET /chain/documents/{document_id}/status`
- `GET /chain/documents/{document_id}/view-url` returns a five-minute S3 URL
- `GET /chain/documents/{document_id}/chunks/{chunk_id}/preview`

All session, message, trace, document, preview, and deletion routes enforce authenticated ownership.
Cross-user lookups return `404`.

## Current boundaries

- PDF preview is native; other formats use escaped extracted-text previews.
- Scanned PDFs and image-only documents still require OCR and are rejected.
- Contradiction detection, chart/image interpretation, malware scanning, email invitations, and
  Office-to-PDF conversion are not included.
- Existing anonymous Redis sessions are not reassigned to accounts and expire under their current
  TTL.
