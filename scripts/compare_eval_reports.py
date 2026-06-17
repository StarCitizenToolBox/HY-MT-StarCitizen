import argparse
import json
from pathlib import Path


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two eval_terms.py JSON reports.")
    parser.add_argument("--reference", required=True, help="Reference report, usually LoRA/bf16.")
    parser.add_argument("--candidate", required=True, help="Candidate report, usually quantized ONNX.")
    parser.add_argument("--print-diffs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = load_report(Path(args.reference))
    candidate = load_report(Path(args.candidate))
    reference_rows = {row["case"]["key"]: row for row in reference["rows"]}
    candidate_rows = {row["case"]["key"]: row for row in candidate["rows"]}
    keys = sorted(set(reference_rows) & set(candidate_rows))
    if not keys:
        raise SystemExit("No shared case keys between reports.")

    changed = []
    term_regressions = []
    term_improvements = []
    exact_regressions = []
    for key in keys:
        ref = reference_rows[key]
        cand = candidate_rows[key]
        if ref["output"].strip() != cand["output"].strip():
            changed.append((key, ref, cand))
        if ref["term_hit"] and not cand["term_hit"]:
            term_regressions.append((key, ref, cand))
        if not ref["term_hit"] and cand["term_hit"]:
            term_improvements.append((key, ref, cand))
        if ref["exact_hit"] and not cand["exact_hit"]:
            exact_regressions.append((key, ref, cand))

    summary = {
        "reference": reference["summary"],
        "candidate": candidate["summary"],
        "shared_cases": len(keys),
        "output_change_rate": len(changed) / len(keys),
        "term_accuracy_delta": candidate["summary"]["term_accuracy"] - reference["summary"]["term_accuracy"],
        "exact_accuracy_delta": candidate["summary"]["exact_accuracy"] - reference["summary"]["exact_accuracy"],
        "term_regressions": len(term_regressions),
        "term_improvements": len(term_improvements),
        "exact_regressions": len(exact_regressions),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for key, ref, cand in term_regressions[: args.print_diffs]:
        case = ref["case"]
        print(f"TERM_REGRESSION {key}: {case['source']}")
        print(f"  ref : {ref['output']}")
        print(f"  cand: {cand['output']}")


if __name__ == "__main__":
    main()
