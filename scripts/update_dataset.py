import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hymt_sc.data import (  # noqa: E402
    EN_URL,
    SCWEB_REPO,
    ZH_URL,
    build_example_samples,
    build_dialogue_context_samples,
    build_pairs,
    build_scweb_pairs,
    build_term_contrast_samples,
    build_term_context_samples,
    build_term_samples,
    derive_term_entries,
    download,
    load_term_entries,
    merge_term_entries,
    sample_to_record,
    split_samples,
    update_git_repo,
    write_jsonl,
    write_metadata,
    write_terms_tsv,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Star Citizen zh/en dataset from upstream global.ini files.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=240)
    parser.add_argument("--require-placeholder-match", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-scweb", action="store_true")
    parser.add_argument("--scweb-dir", default="data/raw/ScWeb_Chinese_Translate")
    parser.add_argument("--terms-file", default="data/terms.zh-en.tsv")
    parser.add_argument("--max-mined-terms", type=int, default=800)
    parser.add_argument("--term-repeat", type=int, default=12)
    parser.add_argument("--term-seed-repeat-multiplier", type=int, default=6)
    parser.add_argument("--term-context-repeat", type=int, default=2)
    parser.add_argument("--max-term-context-samples", type=int, default=40000)
    parser.add_argument("--dialogue-context-repeat", type=int, default=1)
    parser.add_argument("--max-dialogue-context-samples", type=int, default=80000)
    parser.add_argument("--dialogue-context-max-vehicles", type=int, default=260)
    parser.add_argument("--dialogue-context-max-locations", type=int, default=120)
    parser.add_argument("--term-contrast-repeat", type=int, default=1)
    parser.add_argument("--max-term-contrast-pairs-per-category", type=int, default=400)
    parser.add_argument("--examples-file", default="data/term_examples.zh-en.tsv")
    parser.add_argument("--example-repeat", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = ROOT / args.raw_dir
    processed_dir = ROOT / args.processed_dir
    zh_path = raw_dir / "global.zh-CN.ini"
    en_path = raw_dir / "global.en.ini"
    scweb_dir = ROOT / args.scweb_dir
    terms_path = ROOT / args.terms_file
    examples_path = ROOT / args.examples_file
    term_entries = load_term_entries(terms_path)

    if not args.skip_download:
        print(f"Downloading zh: {ZH_URL}")
        download(ZH_URL, zh_path)
        print(f"Downloading en: {EN_URL}")
        download(EN_URL, en_path)
        if not args.skip_scweb:
            print(f"Updating ScWeb: {SCWEB_REPO}")
            update_git_repo(SCWEB_REPO, scweb_dir)

    samples, stats = build_pairs(
        en_path=en_path,
        zh_path=zh_path,
        min_len=args.min_len,
        max_len=args.max_len,
        require_placeholder_match=args.require_placeholder_match,
    )
    mined_terms, mined_stats = derive_term_entries(samples, max_terms=args.max_mined_terms)
    term_entries = merge_term_entries(term_entries, mined_terms)
    stats.update(mined_stats)
    stats["term.total"] = len(term_entries)
    if not args.skip_scweb:
        scweb_samples, scweb_stats = build_scweb_pairs(
            scweb_dir,
            min_len=args.min_len,
            max_len=args.max_len,
            term_entries=term_entries,
        )
        samples.extend(scweb_samples)
        stats.update(scweb_stats)
    term_samples, term_stats = build_term_samples(
        term_entries,
        repeat=args.term_repeat,
        seed_repeat_multiplier=args.term_seed_repeat_multiplier,
    )
    samples.extend(term_samples)
    stats.update(term_stats)
    term_context_samples, term_context_stats = build_term_context_samples(
        term_entries,
        repeat=args.term_context_repeat,
        max_samples=args.max_term_context_samples,
    )
    samples.extend(term_context_samples)
    stats.update(term_context_stats)
    dialogue_samples, dialogue_stats = build_dialogue_context_samples(
        term_entries,
        repeat=args.dialogue_context_repeat,
        max_samples=args.max_dialogue_context_samples,
        max_vehicles=args.dialogue_context_max_vehicles,
        max_locations=args.dialogue_context_max_locations,
    )
    samples.extend(dialogue_samples)
    stats.update(dialogue_stats)
    term_contrast_samples, term_contrast_stats = build_term_contrast_samples(
        term_entries,
        repeat=args.term_contrast_repeat,
        max_pairs_per_category=args.max_term_contrast_pairs_per_category,
    )
    samples.extend(term_contrast_samples)
    stats.update(term_contrast_stats)
    example_samples, example_stats = build_example_samples(examples_path, repeat=args.example_repeat)
    samples.extend(example_samples)
    stats.update(example_stats)

    train_samples, eval_samples = split_samples(samples, args.eval_ratio, args.seed)

    counts = {}
    counts["train.zh-en"] = write_jsonl(
        processed_dir / "train.zh-en.jsonl", (sample_to_record(s, "zh-en") for s in train_samples)
    )
    counts["eval.zh-en"] = write_jsonl(
        processed_dir / "eval.zh-en.jsonl", (sample_to_record(s, "zh-en") for s in eval_samples)
    )
    counts["train.en-zh"] = write_jsonl(
        processed_dir / "train.en-zh.jsonl", (sample_to_record(s, "en-zh") for s in train_samples)
    )
    counts["eval.en-zh"] = write_jsonl(
        processed_dir / "eval.en-zh.jsonl", (sample_to_record(s, "en-zh") for s in eval_samples)
    )
    counts["train.mixed"] = write_jsonl(
        processed_dir / "train.mixed.jsonl",
        (
            record
            for s in train_samples
            for record in (sample_to_record(s, "zh-en"), sample_to_record(s, "en-zh"))
        ),
    )
    counts["eval.mixed"] = write_jsonl(
        processed_dir / "eval.mixed.jsonl",
        (
            record
            for s in eval_samples
            for record in (sample_to_record(s, "zh-en"), sample_to_record(s, "en-zh"))
        ),
    )
    counts["pairs.tsv"] = write_tsv(processed_dir / "pairs.tsv", samples)
    counts["terms.merged"] = write_terms_tsv(processed_dir / "terms.merged.zh-en.tsv", term_entries)
    write_metadata(processed_dir / "metadata.json", samples, stats, counts)

    print("Dataset updated.")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    for key, value in counts.items():
        print(f"  wrote {key}: {value}")


if __name__ == "__main__":
    main()
