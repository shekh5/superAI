from reasoning_chain.evidence import verify_claim_evidence
from reasoning_chain.schemas import AnswerClaim

PASSAGE = {
    "chunk_id": "doc-1:c1",
    "document_id": "doc-1",
    "filename": "report.pdf",
    "document_type": "pdf",
    "location": "page 3",
    "page": 3,
    "citation": "[report.pdf, page 3]",
    "text": "Revenue increased by eighteen percent.",
    "retrieval_score": 0.02,
}


def test_claim_with_owned_semantically_matching_chunk_is_supported():
    claims = [
        AnswerClaim(
            text="Revenue increased by eighteen percent.",
            evidence_chunk_ids=["doc-1:c1"],
        )
    ]
    verified, evidence = verify_claim_evidence(
        claims,
        [PASSAGE],
        query_embedder=lambda _: [1.0, 0.0],
        passage_embedder=lambda _: [[1.0, 0.0]],
        threshold=0.70,
    )
    assert verified[0]["status"] == "supported"
    assert verified[0]["support_score"] == 1.0
    assert evidence[0]["chunk_id"] == "doc-1:c1"


def test_foreign_or_unretrieved_chunk_cannot_become_evidence():
    claims = [AnswerClaim(text="Unsupported claim", evidence_chunk_ids=["other:c1"])]
    verified, evidence = verify_claim_evidence(
        claims,
        [PASSAGE],
        query_embedder=lambda _: [1.0, 0.0],
        passage_embedder=lambda _: [[1.0, 0.0]],
    )
    assert verified[0]["status"] == "evidence_missing"
    assert verified[0]["evidence_chunk_ids"] == []
    assert evidence == []


def test_low_similarity_is_not_reported_as_supported():
    claims = [AnswerClaim(text="Costs fell.", evidence_chunk_ids=["doc-1:c1"])]
    verified, _ = verify_claim_evidence(
        claims,
        [PASSAGE],
        query_embedder=lambda _: [1.0, 0.0],
        passage_embedder=lambda _: [[0.0, 1.0]],
        threshold=0.70,
    )
    assert verified[0]["status"] == "evidence_missing"
