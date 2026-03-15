from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from forensics_agent.schemas import ForensicReport


ACCENT = colors.HexColor("#173F5F")
MUTED = colors.HexColor("#5D6D7E")
LIGHT = colors.HexColor("#EAF2F8")
CITE_CLR = "#1A6B9A"


def _styles():
    s = getSampleStyleSheet()
    s.add(
        ParagraphStyle(
            name="CaseTitle",
            parent=s["Title"],
            fontSize=22,
            leading=28,
            textColor=ACCENT,
            spaceAfter=12,
        )
    )
    s.add(
        ParagraphStyle(
            name="SectionHeader",
            parent=s["Heading2"],
            fontSize=13,
            leading=16,
            textColor=ACCENT,
            spaceBefore=14,
            spaceAfter=6,
        )
    )
    s.add(
        ParagraphStyle(
            name="SmallMuted",
            parent=s["BodyText"],
            fontSize=9,
            leading=12,
            textColor=MUTED,
        )
    )
    s.add(
        ParagraphStyle(
            name="TableCell",
            parent=s["BodyText"],
            fontSize=10,
            leading=13,
            wordWrap="CJK",
        )
    )
    s.add(
        ParagraphStyle(
            name="TableHeader",
            parent=s["BodyText"],
            fontSize=10,
            leading=13,
            fontName="Helvetica-Bold",
        )
    )
    s.add(
        ParagraphStyle(
            name="EvidenceEntry",
            parent=s["BodyText"],
            fontSize=9,
            leading=12,
            leftIndent=6,
            textColor=MUTED,
        )
    )
    s["BodyText"].fontSize = 10
    s["BodyText"].leading = 14
    return s


# Helpers
def _safe(text: str) -> str:
    """Escape XML special characters for ReportLab Paragraph markup."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_citations(text: str) -> str:
    """Convert [N] markers to coloured bold spans for ReportLab markup."""
    return re.sub(
        r"\[(\d+)\]",
        rf'<font color="{CITE_CLR}"><b>[\1]</b></font>',
        text,
    )


def _body(text: str) -> str:
    """Escape then render citation markers; convert newlines to breaks."""
    return _render_citations(_safe(text)).replace("\n", "<br/>")


def _kv_table(rows, styles, label_width=1.9, value_width=4.8):
    """
    Build a key-value table where value cells contain Paragraph objects
    so that long text is wrapped correctly.

    The first row is treated as a header row with a shaded background.
    """
    para_rows = []
    for i, (label, value) in enumerate(rows):
        style = styles["TableHeader"] if i == 0 else styles["TableCell"]
        # Wrap both cells in Paragraph so text wraps within the column
        para_rows.append(
            [
                Paragraph(_safe(str(label)), style),
                Paragraph(_safe(str(value)), style),
            ]
        )

    tbl = Table(para_rows, colWidths=[label_width * inch, value_width * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D3DD")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D6E2EA")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


_ERROR_HINTS = (
    "unavailable",
    "search failed",
    "not installed",
    "rate limit",
    "http error",
    "status code",
    "connection error",
    "timed out",
    "duckduckgo",
    "tavily",
    "api key",
    "error:",
    "exception",
)


def _is_error_item(item) -> bool:
    s = (
        (item.summary if hasattr(item, "summary") else item.get("summary", "")) or ""
    ).lower()
    t = (
        (item.title if hasattr(item, "title") else item.get("title", "")) or ""
    ).lower()
    return any(h in s or h in t for h in _ERROR_HINTS)


# PDF generation
def generate_forensic_pdf(report: ForensicReport, output_path: str | Path) -> Path:
    styles = _styles()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=f"Forensic Report — {report.case_id}",
    )

    story = []

    # Header
    story.append(Paragraph("Digital Forensics Report", styles["CaseTitle"]))
    story.append(
        Paragraph(f"Case ID: <b>{_safe(report.case_id)}</b>", styles["BodyText"])
    )
    story.append(Paragraph(f"Image: {_safe(report.image_path)}", styles["SmallMuted"]))
    story.append(Spacer(1, 0.15 * inch))

    # Summary table
    conf_str = (
        f"{report.detector_confidence:.4f}"
        if report.detector_confidence is not None
        else "N/A"
    )
    story.append(
        _kv_table(
            [
                ("Field", "Value"),
                ("Detector verdict", report.detector_verdict),
                ("Detector confidence", conf_str),
                ("Payload family", report.payload_family),
                ("Payload summary", report.payload_summary),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    #  Payload Analysis
    story.append(Paragraph("Payload Analysis", styles["SectionHeader"]))
    # Classification paragraph first
    story.append(Paragraph(_body(report.payload_class_prediction), styles["BodyText"]))
    story.append(Spacer(1, 0.06 * inch))
    # Technical analysis with inline [N] citations
    story.append(Paragraph(_body(report.technical_analysis), styles["BodyText"]))

    # Confidence & Evidence Notes
    story.append(Paragraph("Confidence &amp; Evidence Notes", styles["SectionHeader"]))
    story.append(Paragraph(_body(report.confidence_notes), styles["BodyText"]))

    # Related CVEs
    if report.related_cves:
        story.append(Paragraph("Related CVEs", styles["SectionHeader"]))
        for cve in report.related_cves:
            # CVE ID as bold header
            header = f"<b>{_safe(cve.cve_id)}</b> — {_safe(cve.relevance)}"
            story.append(Paragraph(header, styles["BodyText"]))
            if cve.description:
                story.append(Paragraph(_safe(cve.description), styles["BodyText"]))
            # Show source URLs as clickable-looking text
            for ref_url in cve.references:
                if ref_url:
                    story.append(
                        Paragraph(
                            f"<font color='{CITE_CLR}'>{_safe(ref_url)}</font>",
                            styles["SmallMuted"],
                        )
                    )
            story.append(Spacer(1, 0.08 * inch))

    # Related Research
    if report.similar_work:
        story.append(Paragraph("Related Research", styles["SectionHeader"]))
        for item in report.similar_work:
            if _is_error_item(item):
                continue
            cid = (
                f' <font color="{CITE_CLR}"><b>[{item.citation_id}]</b></font>'
                if item.citation_id
                else ""
            )
            body = f"<b>{_safe(item.title)}</b>{cid}<br/>{_safe(item.summary)}"
            if item.url:
                body += f"<br/><font color='{CITE_CLR}'><i>{_safe(item.url)}</i></font>"
            story.append(Paragraph(body, styles["BodyText"]))
            story.append(Spacer(1, 0.08 * inch))

    # Evidence Log
    # Includes: payload, arxiv results, web results, CVE sources
    # Excludes: error items and the 'similar_work' items already shown above
    shown_titles = {item.title for item in report.similar_work}
    ev_items = [
        e
        for e in report.evidence
        if not _is_error_item(e) and e.title not in shown_titles
    ]
    if ev_items:
        story.append(Paragraph("Evidence Log", styles["SectionHeader"]))
        for item in ev_items:
            cid = f" [{item.citation_id}]" if item.citation_id else ""
            src_label = f"[{item.source}]{cid}"
            body = f"<b>{_safe(src_label)} {_safe(item.title)}</b><br/>{_safe(item.summary)}"
            if item.url:
                body += f"<br/><font color='{CITE_CLR}'><i>{_safe(item.url)}</i></font>"
            story.append(Paragraph(body, styles["EvidenceEntry"]))
            story.append(Spacer(1, 0.06 * inch))

    # Prevent blank last page
    # ReportLab sometimes emits a blank final page when the story ends exactly
    # at a page boundary.  Strip any trailing Spacers to avoid this.
    while story and isinstance(story[-1], Spacer):
        story.pop()

    doc.build(story)
    return output_path


def generate_blank_template(output_path: str | Path) -> Path:
    placeholder = ForensicReport(
        case_id="CASE-000",
        image_path="/path/to/image.png",
        detector_verdict="unknown",
        detector_confidence=0.0,
        payload_family="unknown",
        payload_class_prediction="unknown",
        payload_summary="Placeholder: populate from agent output.",
        technical_analysis="Placeholder: populate from agent output.",
        related_cves=[],
        similar_work=[],
        evidence=[],
        confidence_notes="Placeholder: CNN score, payload token, SHAP summary.",
    )
    return generate_forensic_pdf(placeholder, output_path)
