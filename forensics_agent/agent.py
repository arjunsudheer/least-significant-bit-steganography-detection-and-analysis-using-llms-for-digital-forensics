from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from common.io_utils import ensure_dir, save_json
from payload_extractor.lsb_extractor import extract_lsb_robindavid_text
from forensics_agent.pdf_report import generate_forensic_pdf
from forensics_agent.schemas import EvidenceItem, ForensicReport
from forensics_agent.tools import search_arxiv, search_cve


def _classify_payload(payload: str) -> tuple[str, str, str]:
    """Classify the payload family and return targeted search queries.

    Returns (payload_family, arxiv_query, cve_query).
    """
    p = payload.lower()

    if any(
        k in p
        for k in (
            "invoke-expression",
            "invoke-webrequest",
            "invoke-psimage",
            "powershell",
            "-encodedcommand",
            "-noprofile",
            "bypass",
            "system.reflection",
            "downloadstring",
            ".ps1",
            "iex(",
        )
    ):
        return (
            "PowerShell",
            "PowerShell fileless malware steganography steganalysis detection",
            "PowerShell Invoke-PSImage steganography malware CVE site:nvd.nist.gov OR site:cve.mitre.org",
        )

    if any(k in p for k in ("<html", "<script", "<body", "<!doctype")) and any(
        k in p for k in ("eval(", "atob(", "unescape(", "fromcharcode", "charcode")
    ):
        return (
            "Obfuscated JavaScript in HTML",
            "obfuscated JavaScript HTML steganography malware payload detection",
            "obfuscated JavaScript HTML steganography malware CVE site:nvd.nist.gov OR site:cve.mitre.org",
        )

    if any(
        k in p
        for k in (
            "eval(",
            "atob(",
            "document.write",
            "xmlhttprequest",
            "fetch(",
            ".js",
            "function(",
            "var ",
            "let ",
            "const ",
            "settimeout",
            "setinterval",
        )
    ):
        return (
            "JavaScript",
            "JavaScript steganography payload delivery malware detection",
            "JavaScript steganography malware CVE site:nvd.nist.gov OR site:cve.mitre.org",
        )

    if re.search(r"\b0x[0-9a-fA-F]{40}\b", payload) or any(
        k in p for k in ("ethereum", "wallet", "ether", "gwei", "wei", "erc20", "web3")
    ):
        return (
            "Ethereum address",
            "Ethereum wallet steganography ransomware C2 cryptocurrency detection",
            "Ethereum wallet steganography ransomware CVE site:nvd.nist.gov OR site:cve.mitre.org",
        )

    if re.search(r"https?://", payload) or re.search(
        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", payload
    ):
        return (
            "URL/IP address",
            "C2 URL IP steganography malware exfiltration payload detection",
            "C2 steganography URL IP malware CVE site:nvd.nist.gov OR site:cve.mitre.org",
        )

    return (
        "Unknown",
        "LSB steganography malware payload steganalysis deep learning",
        "LSB steganography malware CVE site:nvd.nist.gov OR site:cve.mitre.org",
    )


def _is_error_result(item: Dict) -> bool:
    """Return True if a search result represents an error or unavailable response."""
    _ERROR_HINTS = (
        "unavailable",
        "search failed",
        "not installed",
        "rate limit",
        "http error",
        "status code",
        "connection error",
        "timed out",
        "api key",
    )
    summary_lower = (item.get("summary") or "").lower()
    title_lower = (item.get("title") or "").lower()
    return any(h in summary_lower or h in title_lower for h in _ERROR_HINTS)


class AgentState(TypedDict, total=False):
    case_id: str
    image_path: str
    extracted_payload: str
    payload_extraction: Dict
    detector_verdict: str
    detector_confidence: float
    evidence: List[Dict]
    report: Dict
    output_dir: str


def node_collect_evidence(state: AgentState) -> AgentState:
    """Collect evidence via two arXiv searches and one CVE web search.

    arXiv results are tagged source='arxiv' and are intended for the similar_work
    section of the report. CVE results are tagged source='web' and are intended
    for the related_cves section. Citation IDs are assigned sequentially across
    all successful results. Failed or error results are silently discarded.
    """
    payload = state.get("extracted_payload", "")
    ratio = state.get("payload_extraction", {}).get("printable_ratio", 0.0)
    readable = bool(payload.strip()) and ratio >= 0.1

    family, arxiv_q, cve_q = (
        _classify_payload(payload)
        if readable
        else (
            "Unknown",
            "LSB steganography steganalysis deep learning malware detection",
            "LSB steganography malware CVE site:nvd.nist.gov OR site:cve.mitre.org",
        )
    )

    evidence: List[Dict] = [
        {
            "source": "payload",
            "title": "Extracted LSB payload",
            "summary": (
                payload[:2000] if readable else "[Unreadable — binary or encrypted]"
            ),
            "url": None,
            "citation_id": None,
        }
    ]

    ctr = 0

    # arXiv search 1: payload family techniques and detection
    try:
        for item in search_arxiv(arxiv_q, max_results=3):
            if _is_error_result(item):
                continue
            ctr += 1
            evidence.append(
                {
                    "source": "arxiv",
                    "title": item["title"],
                    "summary": item["summary"],
                    "url": item.get("url"),
                    "citation_id": ctr,
                }
            )
    except Exception:
        pass

    # arXiv search 2: LSB steganalysis and SRM-based detection methods
    try:
        for item in search_arxiv(
            f"SRM steganalysis LSB {family} deep learning detection", max_results=2
        ):
            if _is_error_result(item):
                continue
            ctr += 1
            evidence.append(
                {
                    "source": "arxiv",
                    "title": item["title"],
                    "summary": item["summary"],
                    "url": item.get("url"),
                    "citation_id": ctr,
                }
            )
    except Exception:
        pass

    # CVE search: known vulnerabilities associated with the detected payload family
    try:
        for item in search_cve(cve_q, max_results=4):
            if _is_error_result(item):
                continue
            ctr += 1
            evidence.append(
                {
                    "source": "web",
                    "title": item["title"],
                    "summary": item["summary"],
                    "url": item.get("url"),
                    "citation_id": ctr,
                }
            )
    except Exception:
        pass

    return {"evidence": evidence}


_SYSTEM_PROMPT = """\
You are a senior digital forensics analyst specialising in image steganography.

The CNN detector uses SRM (Spatial Rich Model) filters that suppress natural image
content and amplify LSB noise residuals, enabling detection of single-bit pixel changes.

Dataset payload classes: Clean | JavaScript | Obfuscated JavaScript in HTML | PowerShell | Ethereum address | URL/IP address

Analysis rules:
1. payload_class_prediction: quote 1-3 specific token strings from the payload as evidence.
2. technical_analysis: explain step-by-step what the payload code does, referencing specific
   function names and tokens. Use [N] inline citations for arXiv items only.
   Explain how the SRM filter bank detected the embedding pattern.
3. similar_work: populate ONLY from evidence items where source='arxiv'. Each entry must
   include its citation_id and a summary of the paper's key findings, not the abstract.
4. related_cves: populate ONLY from evidence items where source='web'. Include only CVEs
   that are explicitly present in the web evidence. Never invent CVE identifiers.
5. confidence_notes: state the exact CNN score and quote one payload token that supports
   the verdict. Keep to 2 short sentences.
""".strip()


def node_generate_report(state: AgentState) -> AgentState:
    """Generate a structured forensic report from the accumulated evidence."""
    llm = ChatOllama(model="ministral-3:3b", temperature=0)
    evidence = state.get("evidence", [])
    payload = state.get("extracted_payload", "")
    ratio = state.get("payload_extraction", {}).get("printable_ratio", 0.0)
    conf = state.get("detector_confidence", 0.0)

    payload_readable = bool(payload.strip()) and ratio >= 0.1
    family, _, _ = (
        _classify_payload(payload) if payload_readable else ("Unknown", "", "")
    )

    payload_block = (
        payload[:3000]
        if payload_readable
        else "[Unreadable — binary or encrypted content]"
    )

    arxiv_items = [
        e for e in evidence if e.get("source") == "arxiv" and e.get("citation_id")
    ]
    web_items = [
        e for e in evidence if e.get("source") == "web" and e.get("citation_id")
    ]

    arxiv_block = (
        "\n".join(
            f"  [{e['citation_id']}] {e['title']}: {e['summary'][:300]}"
            for e in arxiv_items
        )
        or "  (no arXiv results)"
    )

    web_block = (
        "\n".join(
            f"  [{e['citation_id']}] {e['title']}: {e['summary'][:300]}"
            for e in web_items
        )
        or "  (no CVE results)"
    )

    schema_example = json.dumps(
        {
            "case_id": state["case_id"],
            "image_path": state["image_path"],
            "detector_verdict": "stego",
            "detector_confidence": conf,
            "payload_family": family,
            "payload_class_prediction": "FILL: 3-5 sentences quoting specific payload tokens as evidence",
            "payload_summary": "FILL: one sentence describing what the payload does",
            "technical_analysis": "FILL: 5-8 sentences explaining the execution chain step-by-step, use [N] arXiv citations",
            "related_cves": [
                {
                    "cve_id": "CVE-XXXX-XXXXX",
                    "description": "...",
                    "relevance": "...",
                    "references": ["https://..."],
                }
            ],
            "similar_work": [
                {
                    "source": "arxiv",
                    "title": "...",
                    "summary": "KEY FINDINGS: ...",
                    "url": "...",
                    "citation_id": 1,
                }
            ],
            "evidence": [
                {
                    "source": "...",
                    "title": "...",
                    "summary": "...",
                    "url": None,
                    "citation_id": None,
                }
            ],
            "confidence_notes": f"FILL: (1) CNN score {conf:.4f} interpretation. (2) Specific payload token.",
        },
        indent=2,
    )

    user_prompt = f"""\
Generate a forensic report for this steganography case.
Return ONLY a single valid JSON object. No markdown fences. No extra text.

CASE:
  case_id:             {state["case_id"]}
  image_path:          {state["image_path"]}
  detector_confidence: {conf:.6f}

PAYLOAD ({len(payload)} chars, printable ratio {ratio:.1%}):
{payload_block}

ARXIV EVIDENCE — use [N] citations in technical_analysis, populate similar_work from these:
{arxiv_block}

WEB/CVE EVIDENCE — populate related_cves from these only:
{web_block}

ALL EVIDENCE ({len(evidence)} items):
{json.dumps(evidence, indent=2)[:5000]}

REQUIRED OUTPUT STRUCTURE:
{schema_example}

RULES:
- payload_class_prediction MUST quote specific token strings from the payload.
- technical_analysis MUST explain the execution chain and use [N] arXiv citations.
- similar_work MUST be populated ONLY from source='arxiv' items above.
- related_cves MUST be populated ONLY from source='web' items above. Use [] if none found.
- confidence_notes: CNN score {conf:.4f} and one payload token. Two sentences only.
- Return ONLY the JSON. No preamble, no markdown fences.
""".strip()

    report = None

    try:
        structured = llm.with_structured_output(ForensicReport).invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
        )
        report = structured.model_dump()
    except Exception:
        pass

    if report is None:
        try:
            raw = llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = raw.content if hasattr(raw, "content") else str(raw)
            content = re.sub(r"```(?:json)?\s*", "", content).strip()
            start = content.index("{")
            end = content.rindex("}") + 1
            report = ForensicReport(**json.loads(content[start:end])).model_dump()
        except Exception as exc:
            report = ForensicReport(
                case_id=state["case_id"],
                image_path=state["image_path"],
                detector_verdict="stego",
                detector_confidence=conf,
                payload_family=family,
                payload_class_prediction=(
                    f"Heuristic classification: {family}. "
                    + (
                        f"Payload tokens: `{payload[:200]}`"
                        if payload_readable
                        else "Payload is binary or encrypted — classification based on CNN confidence."
                    )
                ),
                payload_summary=f"{family} payload detected via CNN steganalysis.",
                technical_analysis=(
                    f"The CNN detected steganographic content with confidence {conf:.4f} "
                    f"using SRM residual filters. Heuristic analysis identified the payload "
                    f"as {family}. "
                    + (
                        f"Key payload tokens: `{payload[:300]}`."
                        if payload_readable
                        else "Payload is not human-readable."
                    )
                    + f" Report generation error: {exc}"
                ),
                related_cves=[],
                similar_work=[],
                evidence=[
                    EvidenceItem(
                        **{k: v for k, v in e.items() if k in EvidenceItem.model_fields}
                    )
                    for e in evidence
                    if not _is_error_result(e) and e.get("source") != "payload"
                ],
                confidence_notes=(
                    f"CNN confidence {conf:.4f} — high-confidence stego detection. "
                    + (
                        f"Payload token `{payload[:80]}` supports {family} classification."
                        if payload_readable
                        else "Payload is unreadable; classification based on CNN verdict only."
                    )
                ),
            ).model_dump()

    report["detector_verdict"] = state.get("detector_verdict", "stego")
    report["detector_confidence"] = conf

    return {"report": report}


def node_write_outputs(state: AgentState) -> AgentState:
    """Serialise the report to JSON and render it as a PDF."""
    out_dir = ensure_dir(state["output_dir"])
    report = ForensicReport(**state["report"])
    save_json(report.model_dump(), out_dir / "forensic_report.json")
    pdf_path = generate_forensic_pdf(report, out_dir / "forensic_report.pdf")
    return {"pdf_path": str(pdf_path)}


def build_graph():
    """Build and compile the forensics agent LangGraph state machine."""
    graph = StateGraph(AgentState)
    graph.add_node("collect_evidence", node_collect_evidence)
    graph.add_node("generate_report", node_generate_report)
    graph.add_node("write_outputs", node_write_outputs)
    graph.set_entry_point("collect_evidence")
    graph.add_edge("collect_evidence", "generate_report")
    graph.add_edge("generate_report", "write_outputs")
    graph.add_edge("write_outputs", END)
    compiled = graph.compile()

    graph_image = compiled.get_graph().draw_mermaid_png()
    with open("artifacts/graph.png", mode="wb") as f:
        f.write(graph_image)

    return compiled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the forensics agent on a single image."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--extracted-payload", default=None)
    parser.add_argument("--case-id", default="CASE-001")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detector-verdict", default="stego")
    parser.add_argument("--detector-confidence", type=float, default=0.99)
    parser.add_argument("--auto-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.extracted_payload:
        payload = Path(args.extracted_payload).read_text(encoding="utf-8")
    elif args.auto_extract:
        payload = extract_lsb_robindavid_text(args.image)
    else:
        raise ValueError("Provide --extracted-payload or --auto-extract.")

    result = build_graph().invoke(
        {
            "case_id": args.case_id,
            "image_path": args.image,
            "extracted_payload": payload,
            "detector_verdict": args.detector_verdict,
            "detector_confidence": args.detector_confidence,
            "output_dir": args.output_dir,
        }
    )
    print(json.dumps({k: v for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
