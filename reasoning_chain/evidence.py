"""Provider-independent deterministic claim-to-evidence scoring."""

from __future__ import annotations

import math
from collections.abc import Callable

from .workspace_config import workspace_settings


def verify_claim_evidence(
    claims: list,
    passages: list[dict],
    *,
    query_embedder: Callable[[str], list[float]] | None = None,
    passage_embedder: Callable[[list[str]], list[list[float]]] | None = None,
    threshold: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Validate chunk ownership-by-retrieval and score support without another LLM call."""
    has_stored_vectors = all(passage.get("embedding") for passage in passages)
    if query_embedder is None or (passage_embedder is None and not has_stored_vectors):
        from .workspace import embed_passages, embed_query

        query_embedder = query_embedder or embed_query
        passage_embedder = passage_embedder or embed_passages
    threshold = workspace_settings.evidence_threshold if threshold is None else threshold
    passage_by_id = {passage["chunk_id"]: passage for passage in passages}
    evidence_vectors = (
        [passage["embedding"] for passage in passages]
        if has_stored_vectors
        else passage_embedder([passage["text"] for passage in passages])
    )
    vector_by_id = {
        passage["chunk_id"]: vector
        for passage, vector in zip(passages, evidence_vectors)
    }
    verified = []
    used_ids = []
    for claim in claims:
        claim_text = claim.text if hasattr(claim, "text") else str(claim.get("text", ""))
        requested = (
            claim.evidence_chunk_ids
            if hasattr(claim, "evidence_chunk_ids")
            else claim.get("evidence_chunk_ids", [])
        )
        valid_ids = [chunk_id for chunk_id in requested if chunk_id in passage_by_id]
        claim_vector = query_embedder(claim_text) if valid_ids else []
        similarities = []
        for chunk_id in valid_ids:
            evidence_vector = vector_by_id[chunk_id]
            similarities.append(
                sum(a * b for a, b in zip(claim_vector, evidence_vector))
                / max(
                    1e-9,
                    math.sqrt(sum(value * value for value in claim_vector))
                    * math.sqrt(sum(value * value for value in evidence_vector)),
                )
            )
        score = max(similarities, default=0.0)
        supported = bool(valid_ids) and score >= threshold
        verified.append(
            {
                "text": claim_text,
                "status": "supported" if supported else "evidence_missing",
                "support_score": round(score, 3),
                "evidence_chunk_ids": valid_ids,
            }
        )
        used_ids.extend(valid_ids)
    evidence = []
    for chunk_id in dict.fromkeys(used_ids):
        passage = passage_by_id[chunk_id]
        evidence.append(
            {
                "chunk_id": passage["chunk_id"],
                "document_id": passage["document_id"],
                "filename": passage["filename"],
                "document_type": passage.get("document_type", ""),
                "location": passage["location"],
                "page": passage.get("page"),
                "citation": passage["citation"],
                "excerpt": passage["text"][:800],
                "retrieval_score": passage.get("retrieval_score", 0.0),
            }
        )
    return verified, evidence
