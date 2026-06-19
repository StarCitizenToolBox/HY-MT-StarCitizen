import argparse
import hashlib
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_URL = "https://raw.githubusercontent.com/UEETianX/n55-bot/main/n55/src/plugins/ship.py"

SHIP_RULE_PATTERN = re.compile(
    r"^\s*if\s+(?P<condition>.+?):\s*\r?\n\s*path\s*=.*?/ship/(?P<filename>[^'\"]+)",
    re.MULTILINE,
)
QUERY_LITERAL_PATTERN = re.compile(r"['\"](?P<query>查[^'\"]+)['\"]")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
NON_KEY_CHARS = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ShipAlias:
    key: str
    category: str
    en: str
    zh: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Chinese ship aliases from n55-bot ship.py.")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", default="data/ship_aliases.zh-en.tsv")
    return parser.parse_args()


def download_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8-sig", errors="replace")


def contains_cjk(text: str) -> bool:
    return CJK_PATTERN.search(text) is not None


def normalize_alias(query: str) -> str:
    alias = query.strip().replace(" ", "")
    if alias.startswith("查"):
        alias = alias[1:]
    return alias.strip()


def normalize_ship_name(filename: str) -> str:
    stem = Path(urllib.parse.unquote(filename)).stem
    name = stem.replace("_", " ").strip()
    name = re.sub(r"\s*\(replica\)$", "", name, flags=re.IGNORECASE)
    if name.islower():
        return name.title()
    return name


def slug(text: str) -> str:
    value = NON_KEY_CHARS.sub("_", text.casefold()).strip("_")
    return value or "ship"


def alias_id(text: str) -> str:
    value = slug(text)
    if value != "ship":
        return value
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def extract_aliases(source: str, source_url: str) -> tuple[list[ShipAlias], dict[str, int]]:
    aliases: list[ShipAlias] = []
    seen: set[tuple[str, str]] = set()
    zh_to_en: dict[str, str] = {}
    stats = {
        "rules": 0,
        "query_literals": 0,
        "non_chinese_aliases": 0,
        "non_english_names": 0,
        "duplicate_aliases": 0,
        "conflicting_aliases": 0,
    }

    for match in SHIP_RULE_PATTERN.finditer(source):
        stats["rules"] += 1
        en = normalize_ship_name(match.group("filename"))
        if contains_cjk(en):
            stats["non_english_names"] += 1
            continue
        for query_match in QUERY_LITERAL_PATTERN.finditer(match.group("condition")):
            stats["query_literals"] += 1
            zh = normalize_alias(query_match.group("query"))
            if not zh or zh == "飞船" or not contains_cjk(zh):
                stats["non_chinese_aliases"] += 1
                continue
            previous = zh_to_en.get(zh)
            if previous and previous.casefold() != en.casefold():
                stats["conflicting_aliases"] += 1
                continue
            zh_to_en[zh] = en
            dedupe_key = (zh, en.casefold())
            if dedupe_key in seen:
                stats["duplicate_aliases"] += 1
                continue
            seen.add(dedupe_key)
            aliases.append(
                ShipAlias(
                    key=f"ship_alias.{slug(en)}.{alias_id(zh)}",
                    category="vehicle",
                    en=en,
                    zh=zh,
                    source=source_url,
                )
            )

    aliases.sort(key=lambda item: (item.en.casefold(), item.zh))
    stats["aliases"] = len(aliases)
    return aliases, stats


def write_aliases(path: Path, aliases: list[ShipAlias]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("key\tcategory\ten\tzh\tsource\n")
        for alias in aliases:
            f.write(
                "\t".join(
                    [
                        alias.key,
                        alias.category,
                        alias.en.replace("\t", " ").replace("\n", " "),
                        alias.zh.replace("\t", " ").replace("\n", " "),
                        alias.source,
                    ]
                )
                + "\n"
            )
    return len(aliases)


def main() -> None:
    args = parse_args()
    output_path = ROOT / args.output
    source = download_text(args.source_url)
    aliases, stats = extract_aliases(source, args.source_url)
    if not aliases:
        raise RuntimeError(f"No ship aliases extracted from {args.source_url}")
    count = write_aliases(output_path, aliases)
    print(f"Wrote {count} ship aliases to {output_path}")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    sys.exit(main())
