import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hymt_sc.data import (  # noqa: E402
    PairSample,
    build_term_context_samples,
    load_term_entries,
    prompt_for,
)


@dataclass
class EvalCase:
    key: str
    direction: str
    source: str
    target: str
    expected_term: str
    forbidden_terms: list[str]


def resolve_existing_or_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def english_name_present(text: str, english: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(english)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


def term_present(text: str, term: str) -> bool:
    if re.search(r"[A-Za-z]", term):
        return english_name_present(text, term)
    return term in text


def make_cases(args: argparse.Namespace) -> list[EvalCase]:
    terms = load_term_entries(resolve_existing_or_project_path(args.terms_file))
    cases: list[EvalCase] = []

    selected_terms = terms[: args.max_terms] if args.max_terms > 0 else terms
    by_category: dict[str, list[Any]] = {}
    for entry in selected_terms:
        by_category.setdefault(entry.category, []).append(entry)
        cases.append(
            EvalCase(
                key=f"term:{entry.key}",
                direction="zh-en",
                source=entry.zh,
                target=entry.en,
                expected_term=entry.en,
                forbidden_terms=[],
            )
        )

    context_samples, _ = build_term_context_samples(
        selected_terms,
        repeat=1,
        max_samples=args.max_context_cases,
    )
    for sample in context_samples:
        expected = ""
        for entry in selected_terms:
            if entry.zh in sample.zh and term_present(sample.en, entry.en):
                expected = entry.en
                break
        if not expected:
            continue
        cases.append(
            EvalCase(
                key=sample.key,
                direction="zh-en",
                source=sample.zh,
                target=sample.en,
                expected_term=expected,
                forbidden_terms=[],
            )
        )

    examples_path = resolve_existing_or_project_path(args.examples_file)
    if examples_path.exists():
        for line_number, raw_line in enumerate(examples_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line_number == 1 and line.casefold().startswith("key\t"):
                continue
            key, _category, en, zh = raw_line.split("\t")[:4]
            expected = ""
            forbidden_terms: list[str] = []
            for entry in selected_terms:
                if entry.zh in zh and term_present(en, entry.en):
                    expected = entry.en
                elif entry.zh in zh:
                    forbidden_terms.append(entry.en)
            if expected:
                cases.append(
                    EvalCase(
                        key=f"example:{key}",
                        direction="zh-en",
                        source=zh,
                        target=en,
                        expected_term=expected,
                        forbidden_terms=forbidden_terms,
                    )
                )

    for category_entries in by_category.values():
        if len(category_entries) < 2:
            continue
        for left, right in zip(category_entries, category_entries[1:]):
            cases.append(
                EvalCase(
                    key=f"contrast:{left.key}:not:{right.key}",
                    direction="zh-en",
                    source=f"我说的是{left.zh}，不是{right.zh}。",
                    target=f"I said {left.en}, not {right.en}.",
                    expected_term=left.en,
                    forbidden_terms=[],
                )
            )

    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    return cases


class OnnxBackend:
    def __init__(self, args: argparse.Namespace):
        import numpy as np
        from transformers import AutoTokenizer
        from ort_infer import greedy_generate, make_session

        self.np = np
        self.greedy_generate = greedy_generate
        onnx_dir = resolve_existing_or_project_path(args.onnx_dir)
        tokenizer_dir = resolve_existing_or_project_path(args.tokenizer_dir)
        self.session = make_session(str(onnx_dir / args.filename), args.provider)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
        self.max_new_tokens = args.max_new_tokens

    def translate(self, case: EvalCase) -> str:
        prompt = prompt_for(case.direction, case.source)
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = [int(token_id) for token_id in self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]]
        new_ids, _seconds = self.greedy_generate(self.session, input_ids, self.max_new_tokens)
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


class LoraBackend:
    def __init__(self, args: argparse.Namespace):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = args.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model_dir = resolve_existing_or_project_path(args.model_name_or_path)
        adapter_path = resolve_existing_or_project_path(args.adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_path))
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = args.max_new_tokens

    def translate(self, case: EvalCase) -> str:
        prompt = prompt_for(case.direction, case.source)
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def build_backend(name: str, args: argparse.Namespace):
    if name == "onnx":
        return OnnxBackend(args)
    if name == "lora":
        return LoraBackend(args)
    raise ValueError(f"Unsupported backend: {name}")


def score_output(case: EvalCase, output: str) -> dict[str, Any]:
    return {
        "term_hit": term_present(output, case.expected_term),
        "exact_hit": normalize(output) == normalize(case.target),
        "forbidden_hit": any(term_present(output, term) for term in case.forbidden_terms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Star Citizen terminology accuracy.")
    parser.add_argument("--backend", choices=["onnx", "lora"], default="onnx")
    parser.add_argument("--terms-file", default="data/terms.zh-en.tsv")
    parser.add_argument("--examples-file", default="data/term_examples.zh-en.tsv")
    parser.add_argument("--max-terms", type=int, default=120)
    parser.add_argument("--max-context-cases", type=int, default=600)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="")
    parser.add_argument("--print-failures", type=int, default=20)
    parser.add_argument("--onnx-dir", default="outputs/onnx-q4acc4-b128")
    parser.add_argument("--filename", default="model_q4acc4_b128.onnx")
    parser.add_argument("--tokenizer-dir", default="outputs/onnx-q4acc4-b128")
    parser.add_argument("--provider", choices=["cpu", "cuda", "dml"], default="cpu")
    parser.add_argument("--model-name-or-path", default="models/hy-mt2-model")
    parser.add_argument("--adapter-path", default="outputs/hymt-starcitizen-lora")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = make_cases(args)
    backend = build_backend(args.backend, args)
    rows = []
    start = time.time()
    for case in cases:
        output = backend.translate(case)
        score = score_output(case, output)
        rows.append({"case": asdict(case), "output": output, **score})

    total = len(rows)
    term_hits = sum(1 for row in rows if row["term_hit"])
    exact_hits = sum(1 for row in rows if row["exact_hit"])
    forbidden_hits = sum(1 for row in rows if row["forbidden_hit"])
    summary = {
        "backend": args.backend,
        "cases": total,
        "term_accuracy": term_hits / total if total else 0.0,
        "exact_accuracy": exact_hits / total if total else 0.0,
        "forbidden_rate": forbidden_hits / total if total else 0.0,
        "seconds": round(time.time() - start, 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failures = [row for row in rows if not row["term_hit"] or row["forbidden_hit"]]
    for row in failures[: args.print_failures]:
        case = row["case"]
        print(f"FAIL {case['key']}: {case['source']} -> {row['output']} (expected term {case['expected_term']})")

    if args.output:
        output_path = resolve_existing_or_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
