from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    source: str = Field(..., description="Origin: payload | web | arxiv")
    title: str
    summary: str = Field(
        ...,
        description=(
            "For arxiv items: the key findings of the paper relevant to this case. "
            "For web/CVE items: the specific threat-intelligence detail or CVE description."
        ),
    )
    url: Optional[str] = None
    citation_id: Optional[int] = Field(
        default=None,
        description="Sequential integer used as [N] inline citations in technical_analysis.",
    )


class CVEItem(BaseModel):
    cve_id: str = Field(
        ..., description="CVE identifier found in web search results. Never invented."
    )
    description: str
    relevance: str = Field(
        ...,
        description="How this CVE relates to the extracted payload.",
    )
    references: List[str] = Field(
        default_factory=list,
        description="Source URLs where this CVE was found.",
    )


class ForensicReport(BaseModel):
    case_id: str
    image_path: str
    detector_verdict: str
    detector_confidence: Optional[float] = None

    payload_family: str = Field(
        ...,
        description=(
            "Broad payload category: PowerShell | JavaScript | "
            "Obfuscated JavaScript in HTML | Ethereum address | URL/IP address | Unknown."
        ),
    )
    payload_class_prediction: str = Field(
        ...,
        description=(
            "3-5 sentences identifying the specific payload type, "
            "quoting 1-3 tokens from the payload as evidence. "
            "If the payload is unreadable, base the analysis on the CNN confidence score."
        ),
    )
    payload_summary: str = Field(
        ...,
        description="One sentence describing what the payload does and why it is dangerous.",
    )
    technical_analysis: str = Field(
        ...,
        description=(
            "5-8 sentences explaining step-by-step what the payload code does, "
            "referencing specific tokens and functions, describing the execution chain, "
            "explaining how the CNN SRM filter bank detected the embedding, "
            "and using inline [N] citations referencing citation_id values in similar_work."
        ),
    )
    related_cves: List[CVEItem] = Field(
        default_factory=list,
        description=(
            "CVEs sourced exclusively from web search results. "
            "Populate only from evidence items with source='web'. Never invent identifiers."
        ),
    )
    similar_work: List[EvidenceItem] = Field(
        default_factory=list,
        description=(
            "Research papers sourced exclusively from arXiv. "
            "Populate only from evidence items with source='arxiv'. "
            "Each entry must include a citation_id and a summary of key findings."
        ),
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="All evidence items collected during investigation.",
    )
    confidence_notes: str = Field(
        ...,
        description=(
            "Two concise components: "
            "(1) the CNN confidence score and its interpretation; "
            "(2) a specific payload token that supports the classification verdict."
        ),
    )
