import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hymt_sc.data import (  # noqa: E402
    build_quant_focus_samples,
    load_ship_alias_entries,
    load_term_entries,
    sample_to_record,
    split_samples,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a zh-en quantization-focused terminology training set.")
    parser.add_argument("--terms-file", default="data/processed/terms.merged.zh-en.tsv")
    parser.add_argument("--ship-aliases-file", default="data/ship_aliases.zh-en.tsv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--prefix", default="quant-focus")
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--term-repeat", type=int, default=3)
    parser.add_argument("--alias-repeat", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    terms_path = ROOT / args.terms_file
    aliases_path = ROOT / args.ship_aliases_file
    output_dir = ROOT / args.output_dir

    term_entries = load_term_entries(terms_path)
    alias_entries, skipped_existing = load_ship_alias_entries(aliases_path)
    samples, stats = build_quant_focus_samples(
        term_entries,
        alias_entries,
        term_repeat=args.term_repeat,
        alias_repeat=args.alias_repeat,
    )
    train_samples, eval_samples = split_samples(samples, args.eval_ratio, args.seed)

    def records(rows):
        for sample in rows:
            yield sample_to_record(sample, "zh-en")

    counts = {
        "all": write_jsonl(output_dir / f"all.{args.prefix}.zh-en.jsonl", records(samples)),
        "train": write_jsonl(output_dir / f"train.{args.prefix}.zh-en.jsonl", records(train_samples)),
        "eval": write_jsonl(output_dir / f"eval.{args.prefix}.zh-en.jsonl", records(eval_samples)),
    }
    print("Quant-focus dataset updated.")
    print(f"  terms: {len(term_entries)}")
    print(f"  aliases: {len(alias_entries)}")
    print(f"  aliases.skipped_existing: {skipped_existing}")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    for key, value in counts.items():
        print(f"  wrote {key}: {value}")


if __name__ == "__main__":
    main()
