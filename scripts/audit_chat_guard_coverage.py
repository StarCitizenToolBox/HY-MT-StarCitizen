import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SOURCE_PHRASES = {
    "ship_context": ["小刀", "海盗船", "英仙座", "北极星", "天蝎座", "长刀"],
    "gameplay_terms": [
        "软死亡",
        "犯罪等级",
        "红名",
        "打赏金",
        "赏金目标",
        "跑货",
        "护航",
        "医疗信标",
        "医疗救援",
        "地堡任务",
        "量子燃料",
        "申领时间",
        "炮塔位",
        "锁导弹",
        "同步很差",
        "服务信标",
        "合同",
        "采矿",
        "精炼",
        "打捞",
        "残骸",
        "维修",
        "补油",
        "货物网格",
        "牵引光束",
        "机库",
        "电梯",
    ],
    "slang_terms": [
        "打bounty",
        "跑cargo",
        "escort",
        "soft death",
        "锁missile",
        "q油",
        "cs等级",
        "turret位",
        "med beacon",
        "med rescue",
        "bunker",
        "claim timer",
        "desync",
        "bounty目标",
        "red",
        "service beacon",
        "contract",
        "mining",
        "refinery",
        "salvage",
        "wreck",
        "repair",
        "refuel",
        "cargo grid",
        "tractor beam",
        "hangar",
        "elevator",
        "RMC",
    ],
    "chat_noise": [
        ">F7C-S Hornet Ghost",
        "@...",
        "[全局]",
        "[语音]",
        "@队友",
        ">>>",
        "pad 03",
        "OM-1",
        "marker 2",
        "[Global]",
        "[Party]",
        "[Voice]",
        "[Local]",
        "LFG",
        "WTB",
        "WTS",
        "ASAP",
        "plz",
        "o7",
        "有无",
        "有没有一起的",
    ],
    "chat_style": ["sc全局", "队伍", "yy里", "来人", "萌新注意", "报点", "感觉不太行"],
}


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def load_alias_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    pairs = []
    seen = set()
    with path.open(encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line_number == 1 and line.casefold().startswith("key\t"):
                continue
            parts = raw_line.split("\t")
            if len(parts) < 4:
                continue
            _key, _category, en, zh = (part.strip() for part in parts[:4])
            pair = (zh, en.casefold())
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return pairs


def load_term_pairs(path: Path, category_filter: str | None = None) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    pairs = []
    seen = set()
    with path.open(encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line_number == 1 and line.casefold().startswith("key\t"):
                continue
            parts = raw_line.split("\t")
            if len(parts) < 4:
                continue
            _key, category, en, zh = (part.strip() for part in parts[:4])
            if category_filter and category != category_filter:
                continue
            pair = (zh, en.casefold())
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return pairs


def count_alias_coverage(rows: list[dict[str, Any]], pairs: list[tuple[str, str]]) -> dict[str, int]:
    covered = set()
    alias_path_covered = set()
    for row in rows:
        source = row.get("source", "")
        target = row.get("target", "").casefold()
        is_alias_path = row.get("key", "").startswith("quant_focus_alias")
        for pair in pairs:
            zh, en = pair
            if zh in source and en in target:
                covered.add(pair)
                if is_alias_path:
                    alias_path_covered.add(pair)
    return {
        "alias_file_covered_pairs": len(covered),
        "alias_file_alias_path_pairs": len(alias_path_covered),
    }


def count_pair_coverage(rows: list[dict[str, Any]], pairs: list[tuple[str, str]], key_prefix: str) -> int:
    covered = set()
    selected = [row for row in rows if row.get("key", "").startswith(key_prefix)]
    for row in selected:
        source = row.get("source", "")
        target = row.get("target", "").casefold()
        for pair in pairs:
            zh, en = pair
            if zh in source and en in target:
                covered.add(pair)
    return len(covered)


def key_root(key: str) -> str:
    return key.split(":", 1)[0]


def chat_subkey(key: str) -> str:
    parts = key.split(":")
    return parts[1] if len(parts) > 1 and parts[0] == "chat_guard" else ""


def alias_key(key: str) -> str:
    marker = "ship_alias:"
    if marker not in key:
        return ""
    rest = key.split(marker, 1)[1]
    return rest.split(":", 1)[0]


def count_phrases(rows: list[dict[str, Any]], phrases: list[str], prefix: str | None = None) -> dict[str, int]:
    selected = rows if prefix is None else [row for row in rows if row.get("key", "").startswith(prefix)]
    return {phrase: sum(1 for row in selected if phrase in row.get("source", "")) for phrase in phrases}


def build_report(rows: list[dict[str, Any]], aliases_file: Path, terms_file: Path) -> dict[str, Any]:
    alias_pairs = load_alias_pairs(aliases_file)
    vehicle_pairs = load_term_pairs(terms_file, category_filter="vehicle")
    location_pairs = load_term_pairs(terms_file, category_filter="location")
    gameplay_pairs = load_term_pairs(terms_file, category_filter="gameplay")
    root_counts = Counter(key_root(row.get("key", "")) for row in rows)
    chat_counts = Counter(chat_subkey(row.get("key", "")) for row in rows if row.get("key", "").startswith("chat_guard:"))
    target_cjk = [row for row in rows if re.search(r"[\u3400-\u9fff]", row.get("target", ""))]
    alias_chat_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_alias_chat:")]
    alias_slang_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_alias_slang:")]
    alias_social_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_alias_social:")]
    vehicle_comm_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_vehicle_comm:")]
    vehicle_contrast_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_vehicle_contrast:")]
    vehicle_mixed_format_rows = [
        row for row in rows if row.get("key", "").startswith("quant_focus_vehicle_mixed_format:")
    ]
    vehicle_chat_log_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_vehicle_chat_log:")]
    vehicle_social_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_vehicle_social:")]
    location_comm_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_location_comm:")]
    location_route_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_location_route:")]
    location_mixed_format_rows = [
        row for row in rows if row.get("key", "").startswith("quant_focus_location_mixed_format:")
    ]
    location_chat_log_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_location_chat_log:")]
    location_social_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_location_social:")]
    alias_chat_log_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_alias_chat_log:")]
    gameplay_comm_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_gameplay_comm:")]
    gameplay_social_rows = [row for row in rows if row.get("key", "").startswith("quant_focus_gameplay_social:")]
    alias_chat_keys = {alias_key(row.get("key", "")) for row in alias_chat_rows}
    alias_slang_keys = {alias_key(row.get("key", "")) for row in alias_slang_rows}

    phrase_counts = {}
    alias_phrase_counts = {}
    for group, phrases in SOURCE_PHRASES.items():
        phrase_counts[group] = count_phrases(rows, phrases)
        alias_phrase_counts[group] = count_phrases(rows, phrases, prefix="quant_focus_alias")

    return {
        "rows": len(rows),
        "target_cjk_count": len(target_cjk),
        "alias_file_rows": len(alias_pairs),
        "alias_file_unique_pairs": len(set(alias_pairs)),
        **count_alias_coverage(rows, alias_pairs),
        "vehicle_term_rows": len(vehicle_pairs),
        "vehicle_term_unique_pairs": len(set(vehicle_pairs)),
        "vehicle_comm_rows": len(vehicle_comm_rows),
        "vehicle_comm_covered_pairs": count_pair_coverage(rows, vehicle_pairs, "quant_focus_vehicle_comm:"),
        "vehicle_contrast_rows": len(vehicle_contrast_rows),
        "vehicle_contrast_covered_pairs": count_pair_coverage(rows, vehicle_pairs, "quant_focus_vehicle_contrast:"),
        "vehicle_mixed_format_rows": len(vehicle_mixed_format_rows),
        "vehicle_mixed_format_covered_pairs": count_pair_coverage(
            rows,
            vehicle_pairs,
            "quant_focus_vehicle_mixed_format:",
        ),
        "vehicle_chat_log_rows": len(vehicle_chat_log_rows),
        "vehicle_chat_log_covered_pairs": count_pair_coverage(rows, vehicle_pairs, "quant_focus_vehicle_chat_log:"),
        "vehicle_social_rows": len(vehicle_social_rows),
        "vehicle_social_covered_pairs": count_pair_coverage(rows, vehicle_pairs, "quant_focus_vehicle_social:"),
        "location_term_rows": len(location_pairs),
        "location_term_unique_pairs": len(set(location_pairs)),
        "location_comm_rows": len(location_comm_rows),
        "location_comm_covered_pairs": count_pair_coverage(rows, location_pairs, "quant_focus_location_comm:"),
        "location_route_rows": len(location_route_rows),
        "location_route_covered_pairs": count_pair_coverage(rows, location_pairs, "quant_focus_location_route:"),
        "location_mixed_format_rows": len(location_mixed_format_rows),
        "location_mixed_format_covered_pairs": count_pair_coverage(
            rows,
            location_pairs,
            "quant_focus_location_mixed_format:",
        ),
        "location_chat_log_rows": len(location_chat_log_rows),
        "location_chat_log_covered_pairs": count_pair_coverage(rows, location_pairs, "quant_focus_location_chat_log:"),
        "location_social_rows": len(location_social_rows),
        "location_social_covered_pairs": count_pair_coverage(rows, location_pairs, "quant_focus_location_social:"),
        "alias_chat_log_rows": len(alias_chat_log_rows),
        "alias_chat_log_covered_pairs": count_pair_coverage(rows, alias_pairs, "quant_focus_alias_chat_log:"),
        "alias_social_rows": len(alias_social_rows),
        "alias_social_covered_pairs": count_pair_coverage(rows, alias_pairs, "quant_focus_alias_social:"),
        "gameplay_term_rows": len(gameplay_pairs),
        "gameplay_term_unique_pairs": len(set(gameplay_pairs)),
        "gameplay_comm_rows": len(gameplay_comm_rows),
        "gameplay_comm_covered_pairs": count_pair_coverage(rows, gameplay_pairs, "quant_focus_gameplay_comm:"),
        "gameplay_social_rows": len(gameplay_social_rows),
        "gameplay_social_covered_pairs": count_pair_coverage(rows, gameplay_pairs, "quant_focus_gameplay_social:"),
        "root_counts": dict(sorted(root_counts.items())),
        "chat_subkey_counts": dict(sorted((key, value) for key, value in chat_counts.items() if key)),
        "alias_chat_rows": len(alias_chat_rows),
        "alias_slang_rows": len(alias_slang_rows),
        "alias_chat_unique": len(alias_chat_keys - {""}),
        "alias_slang_unique": len(alias_slang_keys - {""}),
        "phrase_counts": phrase_counts,
        "alias_phrase_counts": alias_phrase_counts,
    }


def print_section(title: str, values: dict[str, int]) -> None:
    print(title)
    for key, value in sorted(values.items()):
        print(f"  {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit generated player-chat terminology coverage.")
    parser.add_argument("--input", default="data/processed/all.quant-guard.zh-en.jsonl")
    parser.add_argument("--aliases-file", default="data/ship_aliases.zh-en.tsv")
    parser.add_argument("--terms-file", default="data/processed/terms.merged.zh-en.tsv")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = load_rows(resolve_path(args.input))
    report = build_report(rows, resolve_path(args.aliases_file), resolve_path(args.terms_file))

    print(f"rows: {report['rows']}")
    print(f"target_cjk_count: {report['target_cjk_count']}")
    print(f"alias_file_rows: {report['alias_file_rows']}")
    print(f"alias_file_unique_pairs: {report['alias_file_unique_pairs']}")
    print(f"alias_file_covered_pairs: {report['alias_file_covered_pairs']}")
    print(f"alias_file_alias_path_pairs: {report['alias_file_alias_path_pairs']}")
    print(f"vehicle_term_rows: {report['vehicle_term_rows']}")
    print(f"vehicle_term_unique_pairs: {report['vehicle_term_unique_pairs']}")
    print(f"vehicle_comm_rows: {report['vehicle_comm_rows']}")
    print(f"vehicle_comm_covered_pairs: {report['vehicle_comm_covered_pairs']}")
    print(f"vehicle_contrast_rows: {report['vehicle_contrast_rows']}")
    print(f"vehicle_contrast_covered_pairs: {report['vehicle_contrast_covered_pairs']}")
    print(f"vehicle_mixed_format_rows: {report['vehicle_mixed_format_rows']}")
    print(f"vehicle_mixed_format_covered_pairs: {report['vehicle_mixed_format_covered_pairs']}")
    print(f"vehicle_chat_log_rows: {report['vehicle_chat_log_rows']}")
    print(f"vehicle_chat_log_covered_pairs: {report['vehicle_chat_log_covered_pairs']}")
    print(f"vehicle_social_rows: {report['vehicle_social_rows']}")
    print(f"vehicle_social_covered_pairs: {report['vehicle_social_covered_pairs']}")
    print(f"location_term_rows: {report['location_term_rows']}")
    print(f"location_term_unique_pairs: {report['location_term_unique_pairs']}")
    print(f"location_comm_rows: {report['location_comm_rows']}")
    print(f"location_comm_covered_pairs: {report['location_comm_covered_pairs']}")
    print(f"location_route_rows: {report['location_route_rows']}")
    print(f"location_route_covered_pairs: {report['location_route_covered_pairs']}")
    print(f"location_mixed_format_rows: {report['location_mixed_format_rows']}")
    print(f"location_mixed_format_covered_pairs: {report['location_mixed_format_covered_pairs']}")
    print(f"location_chat_log_rows: {report['location_chat_log_rows']}")
    print(f"location_chat_log_covered_pairs: {report['location_chat_log_covered_pairs']}")
    print(f"location_social_rows: {report['location_social_rows']}")
    print(f"location_social_covered_pairs: {report['location_social_covered_pairs']}")
    print(f"alias_chat_log_rows: {report['alias_chat_log_rows']}")
    print(f"alias_chat_log_covered_pairs: {report['alias_chat_log_covered_pairs']}")
    print(f"alias_social_rows: {report['alias_social_rows']}")
    print(f"alias_social_covered_pairs: {report['alias_social_covered_pairs']}")
    print(f"gameplay_term_rows: {report['gameplay_term_rows']}")
    print(f"gameplay_term_unique_pairs: {report['gameplay_term_unique_pairs']}")
    print(f"gameplay_comm_rows: {report['gameplay_comm_rows']}")
    print(f"gameplay_comm_covered_pairs: {report['gameplay_comm_covered_pairs']}")
    print(f"gameplay_social_rows: {report['gameplay_social_rows']}")
    print(f"gameplay_social_covered_pairs: {report['gameplay_social_covered_pairs']}")
    print(f"alias_chat_rows: {report['alias_chat_rows']}")
    print(f"alias_slang_rows: {report['alias_slang_rows']}")
    print(f"alias_chat_unique: {report['alias_chat_unique']}")
    print(f"alias_slang_unique: {report['alias_slang_unique']}")
    print_section("root_counts:", report["root_counts"])
    print_section("chat_subkey_counts:", report["chat_subkey_counts"])
    for group, values in report["phrase_counts"].items():
        print_section(f"phrase_counts.{group}:", values)
    for group, values in report["alias_phrase_counts"].items():
        print_section(f"alias_phrase_counts.{group}:", values)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
