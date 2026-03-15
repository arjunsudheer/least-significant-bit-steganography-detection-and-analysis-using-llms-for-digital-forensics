from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import List

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate

# Use the non-collections metric classes (ragas.metrics, not ragas.metrics.collections).
# ragas.metrics.collections requires the `embedding_factory` "modern" interface which
# does not support Ollama locally.  The plain ragas.metrics classes accept
# LangchainLLMWrapper / LangchainEmbeddingsWrapper without complaint.
from ragas.metrics import Faithfulness, AnswerRelevancy as _Relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from common.io_utils import ensure_dir


def _build_ragas_llm_and_embeddings(model: str, embed_model: str):
    """
    Build RAGAS-compatible LLM and embedding objects backed by local Ollama.

    Uses LangchainLLMWrapper and LangchainEmbeddingsWrapper (the deprecated but
    still-functional wrappers) because the newer embedding_factory / llm_factory
    APIs do not yet expose a stable path for fully-local Ollama models.
    The DeprecationWarning is suppressed at call-site so it does not pollute logs.
    """
    chat = ChatOllama(model=model, temperature=0)
    ragas_llm = LangchainLLMWrapper(chat)

    try:
        emb_raw = OllamaEmbeddings(model=embed_model)
        emb_raw.embed_query("test")   # probe — raises if model not pulled
        ragas_emb = LangchainEmbeddingsWrapper(emb_raw)
        print(f"[ragas] Embeddings : {embed_model}")
    except Exception:
        print(
            f"[ragas] WARNING: {embed_model} not available; "
            f"falling back to {model} for embeddings (less accurate)."
        )
        ragas_emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=model))

    return ragas_llm, ragas_emb


def _normalise_fields(obj: dict) -> dict:
    """Accept RAGAS 0.1 field names and rename to 0.2 names."""
    renames = {
        "question": "user_input",
        "answer": "response",
        "contexts": "retrieved_contexts",
        "ground_truth": "reference",
        "ground_truths": "reference",
    }
    for old, new in renames.items():
        if old in obj and new not in obj:
            obj[new] = obj.pop(old)
    return obj


def load_cases(jsonl_path: str | Path) -> List[dict]:
    rows = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = _normalise_fields(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i}: {exc}") from exc
            for key in ("user_input", "response", "retrieved_contexts"):
                if key not in obj:
                    raise ValueError(f'Line {i} missing required field "{key}".')
            rows.append(obj)
    if not rows:
        raise ValueError(f"No cases found in {jsonl_path}")
    return rows


_DEFAULT_PROMPT = (
    "You are a digital forensics analyst. "
    "Investigate this steganographic image case and produce a forensic report."
)


def convert_reports_to_ragas_jsonl(
    report_dir: str | Path,
    output_path: str | Path,
    user_input_template: str = _DEFAULT_PROMPT,
) -> int:
    """
    Walk *report_dir* for forensic_report.json files and write RAGAS JSONL.
    Uses the updated schema fields (payload_class_prediction, no recommended_actions).
    Returns the number of rows written.
    """
    report_dir = Path(report_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_files = sorted(report_dir.rglob("forensic_report.json"))
    if not report_files:
        raise FileNotFoundError(f"No forensic_report.json found under {report_dir}")

    rows_written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for rp in report_files:
            try:
                data = json.loads(rp.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[convert] Skipping {rp}: {exc}")
                continue

            response = "\n\n".join(
                filter(
                    None,
                    [
                        f"Case: {data.get('case_id', '')}",
                        f"Payload family: {data.get('payload_family', '')}",
                        f"Payload class prediction: {data.get('payload_class_prediction', '')}",
                        f"Payload summary: {data.get('payload_summary', '')}",
                        f"Technical analysis: {data.get('technical_analysis', '')}",
                        f"Confidence notes: {data.get('confidence_notes', '')}",
                    ],
                )
            )

            contexts = [
                f"[{ev.get('source','')}] {ev.get('title','')}: {ev.get('summary','')}"
                for ev in data.get("evidence", [])
                if ev.get("summary")
                and not any(
                    h in ev.get("summary", "").lower()
                    for h in ("unavailable", "failed", "error", "rate limit")
                )
            ] or [f"Payload summary: {data.get('payload_summary', '')}"]

            row = {
                "user_input": user_input_template,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": "",
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"[convert] Wrote {rows_written} RAGAS cases → {output_path}")
    return rows_written


def run_eval(
    cases_jsonl: str, output_dir_str: str, model: str, embed_model: str
) -> None:
    print(f"[ragas] Judge LLM  : {model}")
    print(f"[ragas] Embed model: {embed_model}")

    rows = load_cases(cases_jsonl)
    print(f"[ragas] Cases      : {len(rows)}")

    dataset = Dataset.from_list(rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ragas_llm, ragas_emb = _build_ragas_llm_and_embeddings(model, embed_model)

    metrics = [
        Faithfulness(llm=ragas_llm),
        _Relevancy(llm=ragas_llm, embeddings=ragas_emb),
    ]

    # Assign local LLM/embeddings to each metric explicitly
    for m in metrics:
        try:
            m.llm = ragas_llm
        except Exception:
            pass
        if hasattr(m, "embeddings"):
            try:
                m.embeddings = ragas_emb
            except Exception:
                pass

    print("[ragas] Evaluating …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_emb,
        )

    out_dir = ensure_dir(output_dir_str)
    df = result.to_pandas() if hasattr(result, "to_pandas") else pd.DataFrame(result)

    csv_path = Path(out_dir) / "ragas_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ragas] Per-case results → {csv_path}")

    def _mean(cols):
        for c in cols:
            if c in df.columns:
                return float(df[c].dropna().mean())
        return None

    summary = {
        "judge_model": model,
        "embed_model": embed_model,
        "num_cases": len(df),
        "faithfulness_mean": _mean(["faithfulness"]),
        "answer_relevancy_mean": _mean(["answer_relevancy", "response_relevancy"]),
    }
    summary_path = Path(out_dir) / "ragas_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[ragas] Summary → {summary_path}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation (fully local via Ollama)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run RAGAS faithfulness + relevancy metrics.")
    p_eval.add_argument(
        "--cases-jsonl", required=True, help="JSONL file of RAGAS-formatted cases."
    )
    p_eval.add_argument(
        "--output-dir",
        required=True,
        help="Directory for ragas_results.csv and ragas_summary.json.",
    )
    p_eval.add_argument(
        "--model", default="ministral-3:3b", help="Ollama chat model used as judge LLM."
    )
    p_eval.add_argument(
        "--embed-model",
        default="nomic-embed-text",
        help="Ollama embedding model. Pull with: ollama pull nomic-embed-text",
    )

    p_conv = sub.add_parser(
        "convert", help="Convert forensic_report.json files to RAGAS JSONL."
    )
    p_conv.add_argument(
        "--report-dir",
        required=True,
        help="Root directory containing forensic_report.json files.",
    )
    p_conv.add_argument("--output", required=True, help="Destination .jsonl file.")

    args = parser.parse_args()

    if args.command == "eval":
        run_eval(args.cases_jsonl, args.output_dir, args.model, args.embed_model)
    elif args.command == "convert":
        convert_reports_to_ragas_jsonl(args.report_dir, args.output)


if __name__ == "__main__":
    main()
