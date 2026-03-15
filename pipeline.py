from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from common.io_utils import ensure_dir, save_json
from payload_extractor.lsb_extractor import extract_payload_text
from binary_classifier.model import SteganalysisNet


_TRANSFORM = T.Compose([
    T.CenterCrop(512),
    T.ToTensor(),
])

_PRINTABLE_THRESHOLD = 0.25


def load_model(model_path: str | Path, device: torch.device) -> SteganalysisNet:
    """Load the steganalysis model in fp16 for memory-efficient inference."""
    net = SteganalysisNet()
    state = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.to(device).half().eval()
    return net


@torch.inference_mode()
def classify_image(
    net: SteganalysisNet,
    image_path: str | Path,
    device: torch.device,
) -> tuple[float, str]:
    """Run the CNN classifier on a single image.

    Returns (sigmoid_probability, label) where label is 'stego' or 'clean'.
    """
    img    = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0)
    if device.type == "cuda":
        tensor = tensor.half()
    prob = torch.sigmoid(net(tensor.to(device))).float().item()
    return prob, ("stego" if prob >= 0.5 else "clean")


def free_activation_cache() -> None:
    """Release cached CUDA activation memory without unloading model weights."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for c in s if c.isprintable() or c in "\n\r\t") / len(s)


def extract_payload(image_path: str | Path) -> tuple[str, float]:
    """Extract the LSB payload and return (payload_text, printable_ratio)."""
    payload = extract_payload_text(image_path)
    ratio   = _printable_ratio(payload)
    return payload, ratio


def run_pipeline(args: argparse.Namespace) -> dict:
    """Run the full steganalysis and forensics pipeline on a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pipeline] Device    : {device}")
    print(f"[pipeline] Image     : {args.image}")

    net     = load_model(args.model_path, device)
    out_dir = ensure_dir(args.output_dir)
    prob, label = classify_image(net, args.image, device)
    print(f"[pipeline] Verdict   : {label.upper()}  (p_stego = {prob:.4f})")

    free_activation_cache()

    if label == "clean":
        result = {
            "case_id":           args.case_id,
            "image_path":        str(args.image),
            "verdict":           "clean",
            "stego_probability": prob,
        }
        save_json(result, out_dir / "verdict.json")
        print(f'[pipeline] Clean. → {out_dir / "verdict.json"}')
        return result

    print("[pipeline] Extracting payload …")
    payload, ratio = extract_payload(args.image)

    if ratio >= _PRINTABLE_THRESHOLD:
        print(f"[pipeline] Payload   : printable={ratio:.1%}, len={len(payload)}")
    else:
        print(f"[pipeline] WARNING   : printable={ratio:.2%} — may be binary/encrypted")

    payload_path = out_dir / "extracted_payload.txt"
    payload_path.write_text(payload, encoding="utf-8", errors="replace")
    preview = payload[:120].replace("\n", " ").replace("\r", "")
    print(f"[pipeline] Preview   : {preview} …")

    if args.no_agent:
        result = {
            "case_id":           args.case_id,
            "image_path":        str(args.image),
            "verdict":           "stego",
            "stego_probability": prob,
            "payload_path":      str(payload_path),
        }
        save_json(result, out_dir / "verdict.json")
        return result

    print("[pipeline] Launching forensics agent …")
    from forensics_agent.agent import build_graph

    result = build_graph().invoke({
        "case_id":             args.case_id,
        "image_path":          str(args.image),
        "extracted_payload":   payload,
        "payload_extraction":  {"printable_ratio": ratio},
        "detector_verdict":    "stego",
        "detector_confidence": prob,
        "output_dir":          str(out_dir),
    })

    summary = {k: v for k, v in result.items() if k != "messages"}
    summary["stego_probability"] = prob
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end steganalysis and forensics pipeline.")
    p.add_argument("--image",      required=True,  help="Path to the input image.")
    p.add_argument("--model-path", required=True,  help="Path to the trained model checkpoint.")
    p.add_argument("--output-dir", required=True,  help="Directory for all outputs.")
    p.add_argument("--case-id",    default="CASE-001")
    p.add_argument("--no-agent",   action="store_true", help="Skip the forensics agent step.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    result = run_pipeline(args)
    print("\n[pipeline] Done.")
    print(json.dumps({k: v for k, v in result.items() if k != "messages"}, indent=2))


if __name__ == "__main__":
    main()
