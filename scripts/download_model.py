import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def resolve_project_output(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the base Hy-MT2 model into this project.")
    parser.add_argument("--repo-id", default="tencent/Hy-MT2-1.8B")
    parser.add_argument("--output-dir", default="models/hy-mt2-model")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_output(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"model_dir={output_dir}")


if __name__ == "__main__":
    main()
