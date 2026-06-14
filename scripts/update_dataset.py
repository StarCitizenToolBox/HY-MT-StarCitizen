import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hymt_sc.data import (  # noqa: E402
    EN_URL,
    SCWEB_REPO,
    ZH_URL,
    build_pairs,
    build_scweb_pairs,
    download,
    sample_to_record,
    split_samples,
    update_git_repo,
    write_jsonl,
    write_metadata,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = ROOT / args.raw_dir
    processed_dir = ROOT / args.processed_dir
    zh_path = raw_dir / "global.zh-CN.ini"
    en_path = raw_dir / "global.en.ini"
    scweb_dir = ROOT / args.scweb_dir

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
    if not args.skip_scweb:
        scweb_samples, scweb_stats = build_scweb_pairs(scweb_dir, min_len=args.min_len, max_len=args.max_len)
        samples.extend(scweb_samples)
        stats.update(scweb_stats)

    train_samples, eval_samples = split_samples(samples, args.eval_ratio, args.seed)

    counts = {}
    counts["train.zh-en"] = write_jsonl(processed_dir / "train.zh-en.jsonl", (sample_to_record(s, "zh-en") for s in train_samples))
    counts["eval.zh-en"] = write_jsonl(processed_dir / "eval.zh-en.jsonl", (sample_to_record(s, "zh-en") for s in eval_samples))
    counts["train.en-zh"] = write_jsonl(processed_dir / "train.en-zh.jsonl", (sample_to_record(s, "en-zh") for s in train_samples))
    counts["eval.en-zh"] = write_jsonl(processed_dir / "eval.en-zh.jsonl", (sample_to_record(s, "en-zh") for s in eval_samples))
    counts["train.mixed"] = write_jsonl(
        processed_dir / "train.mixed.jsonl",
        (record for s in train_samples for record in (sample_to_record(s, "zh-en"), sample_to_record(s, "en-zh"))),
    )
    counts["eval.mixed"] = write_jsonl(
        processed_dir / "eval.mixed.jsonl",
        (record for s in eval_samples for record in (sample_to_record(s, "zh-en"), sample_to_record(s, "en-zh"))),
    )
    counts["pairs.tsv"] = write_tsv(processed_dir / "pairs.tsv", samples)
    write_metadata(processed_dir / "metadata.json", samples, stats, counts)

    print("Dataset updated.")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    for key, value in counts.items():
        print(f"  wrote {key}: {value}")


if __name__ == "__main__":
    main()
