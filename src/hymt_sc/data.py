import json
import random
import re
import subprocess
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .classifier import StarCitizenKeyClassifier


ZH_URL = "https://raw.githubusercontent.com/StarCitizenToolBox/LocalizationData/main/chinese_(simplified)/global.ini"
EN_URL = "https://raw.githubusercontent.com/Dymerz/StarCitizen-Localization/main/data/Localization/english/global.ini"
SCWEB_REPO = "https://github.com/CxJuice/ScWeb_Chinese_Translate"


COMPLEX_PLACEHOLDER_PATTERNS = [
    r"~mission\([^)]+\)",
    r"~\w+\([^)]+\)",
    r"\{[^}]*\|[^}]*\}",
    r"<%[^>]+%>",
]
PLACEHOLDER_PATTERN = re.compile(
    r"%[a-zA-Z_][\w]*|%[0-9.]*[sdif]|~[a-zA-Z_][\w]*(?:\([^)]+\))?|\$[a-zA-Z_][\w]*|@[a-zA-Z_][\w]*|\{[^}]+\}"
)


@dataclass
class PairSample:
    key: str
    en: str
    zh: str
    category: str
    is_priority: bool
    source: str = "global_ini"


def download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        output_path.write_bytes(response.read())


def update_git_repo(repo_url: str, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if (output_dir / ".git").exists():
        subprocess.run(["git", "-C", str(output_dir), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(output_dir)], check=True)


def parse_ini(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";", "//")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = clean_text(value)
        data[key] = value
    return data


def clean_text(text: str) -> str:
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return text.strip()


def placeholders(text: str) -> list[str]:
    return sorted(PLACEHOLDER_PATTERN.findall(text))


def has_complex_placeholder(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in COMPLEX_PLACEHOLDER_PATTERNS)


def is_mostly_markup_or_number(text: str) -> bool:
    stripped = re.sub(r"[\s\d._:：/\\\-+()[\]{}<>|,，。！？!?%]+", "", text)
    return len(stripped) == 0


def valid_pair(en: str, zh: str, min_len: int, max_len: int, require_placeholder_match: bool) -> bool:
    if not en or not zh:
        return False
    if len(en) < min_len or len(zh) < min_len:
        return False
    if len(en) > max_len or len(zh) > max_len:
        return False
    if is_mostly_markup_or_number(en) or is_mostly_markup_or_number(zh):
        return False
    if has_complex_placeholder(en) or has_complex_placeholder(zh):
        return False
    if require_placeholder_match and placeholders(en) != placeholders(zh):
        return False
    return True


def build_pairs(
    en_path: Path,
    zh_path: Path,
    min_len: int = 2,
    max_len: int = 240,
    require_placeholder_match: bool = False,
) -> tuple[list[PairSample], dict[str, int]]:
    en_data = parse_ini(en_path)
    zh_data = parse_ini(zh_path)
    common_keys = sorted(set(en_data) & set(zh_data))
    stats = Counter()
    samples: list[PairSample] = []

    for key in common_keys:
        en = clean_text(en_data[key])
        zh = clean_text(zh_data[key])
        if not valid_pair(en, zh, min_len, max_len, require_placeholder_match):
            stats["filtered"] += 1
            continue
        category = StarCitizenKeyClassifier.classify(key)
        samples.append(
            PairSample(
                key=key,
                en=en,
                zh=zh,
                category=category.category,
                is_priority=category.is_priority,
                source="global_ini",
            )
        )
        stats[f"category.{category.category}"] += 1
        if category.is_priority:
            stats["priority"] += 1
        else:
            stats["other"] += 1

    stats["en_keys"] = len(en_data)
    stats["zh_keys"] = len(zh_data)
    stats["common_keys"] = len(common_keys)
    stats["samples"] = len(samples)
    return samples, dict(stats)


def iter_scweb_json_files(scweb_dir: Path) -> Iterable[Path]:
    locales = scweb_dir / "json" / "locales"
    if not locales.exists():
        return []
    return sorted(path for path in locales.glob("*.json") if path.name != "versions.json")


def build_scweb_pairs(scweb_dir: Path, min_len: int = 2, max_len: int = 240) -> tuple[list[PairSample], dict[str, int]]:
    samples: list[PairSample] = []
    stats = Counter()
    seen: set[tuple[str, str]] = set()

    for path in iter_scweb_json_files(scweb_dir):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            stats["scweb.skipped_non_dict"] += 1
            continue
        for en_raw, zh_raw in data.items():
            en = clean_text(str(en_raw))
            zh = clean_text(str(zh_raw))
            if not valid_pair(en, zh, min_len, max_len, require_placeholder_match=False):
                stats["scweb.filtered"] += 1
                continue
            dedupe_key = (en.casefold(), zh)
            if dedupe_key in seen:
                stats["scweb.duplicates"] += 1
                continue
            seen.add(dedupe_key)
            samples.append(
                PairSample(
                    key=f"scweb:{path.stem}:{en[:80]}",
                    en=en,
                    zh=zh,
                    category="scweb",
                    is_priority=False,
                    source="scweb",
                )
            )

    stats["scweb.samples"] = len(samples)
    return samples, dict(stats)


def prompt_for(direction: str, text: str) -> str:
    if direction == "zh-en":
        return f"Translate the following Star Citizen localization text into English. Only output the translation:\n\n{text}"
    if direction == "en-zh":
        return f"将以下《星际公民》本地化文本翻译为简体中文。只输出翻译结果：\n\n{text}"
    raise ValueError(f"Unsupported direction: {direction}")


def sample_to_record(sample: PairSample, direction: str) -> dict:
    if direction == "zh-en":
        source, target = sample.zh, sample.en
    elif direction == "en-zh":
        source, target = sample.en, sample.zh
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return {
        "key": sample.key,
        "direction": direction,
        "category": sample.category,
        "is_priority": sample.is_priority,
        "source_type": sample.source,
        "source": source,
        "target": target,
        "messages": [
            {"role": "user", "content": prompt_for(direction, source)},
            {"role": "assistant", "content": target},
        ],
    }


def split_samples(samples: list[PairSample], eval_ratio: float, seed: int) -> tuple[list[PairSample], list[PairSample]]:
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio)) if shuffled else 0
    return shuffled[eval_size:], shuffled[:eval_size]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_metadata(path: Path, samples: list[PairSample], stats: dict[str, int], counts: dict[str, int]) -> None:
    category_counts = Counter(sample.category for sample in samples)
    source_counts = Counter(sample.source for sample in samples)
    metadata = {
        "sources": {"en": EN_URL, "zh": ZH_URL, "scweb": SCWEB_REPO},
        "stats": stats,
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "output_counts": dict(sorted(counts.items())),
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def write_tsv(path: Path, samples: Iterable[PairSample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("key\tsource\tcategory\tis_priority\ten\tzh\n")
        for sample in samples:
            row = asdict(sample)
            f.write(
                "\t".join(
                    [
                        row["key"],
                        row["source"],
                        row["category"],
                        str(row["is_priority"]).lower(),
                        row["en"].replace("\t", " ").replace("\n", " "),
                        row["zh"].replace("\t", " ").replace("\n", " "),
                    ]
                )
                + "\n"
            )
            count += 1
    return count
