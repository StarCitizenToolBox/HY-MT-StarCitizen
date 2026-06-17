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


@dataclass(frozen=True)
class TermEntry:
    key: str
    category: str
    en: str
    zh: str


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


def build_scweb_pairs(
    scweb_dir: Path,
    min_len: int = 2,
    max_len: int = 240,
    term_entries: Iterable[TermEntry] | None = None,
) -> tuple[list[PairSample], dict[str, int]]:
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
            if term_entries and has_term_conflict(en, zh, term_entries):
                stats["scweb.term_conflict"] += 1
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


def load_term_entries(path: Path) -> list[TermEntry]:
    if not path.exists():
        return []
    entries: list[TermEntry] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line_number == 1 and line.casefold().startswith("key\t"):
            continue
        parts = raw_line.split("\t")
        if len(parts) < 4:
            raise ValueError(f"Invalid term row {path}:{line_number}; expected key, category, en, zh")
        key, category, en, zh = (clean_text(part) for part in parts[:4])
        entries.append(TermEntry(key=key, category=category, en=en, zh=zh))
    return entries


def build_term_samples(
    entries: Iterable[TermEntry],
    repeat: int = 1,
    seed_repeat_multiplier: int = 1,
) -> tuple[list[PairSample], dict[str, int]]:
    samples: list[PairSample] = []
    for entry in entries:
        entry_repeat = max(1, repeat)
        if not entry.key.startswith("mined:"):
            entry_repeat *= max(1, seed_repeat_multiplier)
        for index in range(entry_repeat):
            samples.append(
                PairSample(
                    key=f"term:{entry.key}:{index + 1}",
                    en=entry.en,
                    zh=entry.zh,
                    category=entry.category,
                    is_priority=True,
                    source="term",
                )
            )
    return samples, {"term.samples": len(samples)}


def build_example_samples(path: Path, repeat: int = 1) -> tuple[list[PairSample], dict[str, int]]:
    if not path.exists():
        return [], {"example.samples": 0}
    samples: list[PairSample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line_number == 1 and line.casefold().startswith("key\t"):
            continue
        parts = raw_line.split("\t")
        if len(parts) < 4:
            raise ValueError(f"Invalid example row {path}:{line_number}; expected key, category, en, zh")
        key, category, en, zh = (clean_text(part) for part in parts[:4])
        for index in range(max(1, repeat)):
            samples.append(
                PairSample(
                    key=f"example:{key}:{index + 1}",
                    en=en,
                    zh=zh,
                    category=category,
                    is_priority=True,
                    source="example",
                )
            )
    return samples, {"example.samples": len(samples)}


def build_ship_alias_samples(
    path: Path,
    repeat: int = 1,
    existing_pairs: Iterable[tuple[str, str]] | None = None,
) -> tuple[list[PairSample], dict[str, int]]:
    if not path.exists():
        return [], {"ship_alias.samples": 0, "ship_alias.skipped_existing": 0}
    samples: list[PairSample] = []
    existing = {(clean_text(zh), clean_text(en).casefold()) for zh, en in (existing_pairs or [])}
    skipped_existing = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line_number == 1 and line.casefold().startswith("key\t"):
            continue
        parts = raw_line.split("\t")
        if len(parts) < 4:
            raise ValueError(f"Invalid ship alias row {path}:{line_number}; expected key, category, en, zh")
        key, category, en, zh = (clean_text(part) for part in parts[:4])
        if not contains_cjk(zh):
            raise ValueError(f"Invalid ship alias row {path}:{line_number}; zh alias must contain CJK text")
        if (zh, en.casefold()) in existing:
            skipped_existing += 1
            continue
        for index in range(max(1, repeat)):
            samples.append(
                PairSample(
                    key=f"ship_alias:{key}:{index + 1}",
                    en=en,
                    zh=zh,
                    category=category,
                    is_priority=True,
                    source="ship_alias",
                )
            )
    return samples, {"ship_alias.samples": len(samples), "ship_alias.skipped_existing": skipped_existing}


def contains_cjk(text: str) -> bool:
    return re.search(r"[\u3400-\u9fff]", text) is not None


def canonical_term_en(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^the\s+", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"[A-Z0-9][A-Z0-9 '\-]+", text):
        return text.title()
    return text


def is_short_term_pair(sample: PairSample) -> bool:
    if not contains_cjk(sample.zh):
        return False
    if len(sample.en) > 48 or len(sample.zh) > 32:
        return False
    if re.search(r"[.!?。！？]", sample.en + sample.zh):
        return False
    if placeholders(sample.en) or placeholders(sample.zh):
        return False
    return True


MINED_TERM_DENYLIST = {
    "battery",
    "aligned",
    "cargo",
    "clinic",
    "core",
    "craft",
    "entry",
    "error",
    "fabricate",
    "filled",
    "floor",
    "full",
    "ground floor",
    "jump",
    "location",
    "medic",
    "missing",
    "personnel",
    "print",
    "slot",
    "standby",
    "success",
    "tractor",
}


def is_mineable_location_key(key: str, en: str) -> bool:
    if re.search(
        r"(?:_title|_obj|_marker|_short|_Create|_Select|Status|Error|^ui_|^salvage_|^mg_|^CFP_|^NTLockdown|^refinery|^outpost_|Outpost_CleanUp)",
        key,
        re.IGNORECASE,
    ):
        return False
    if re.match(
        r"^(?:Investigate|Search|Get|Raid|Deliver|Return|Gather|Hit|Provide|Select|Rectify|Open|A worried)\b",
        en,
        re.IGNORECASE,
    ):
        return False
    if re.fullmatch(r"(?:Floor|City Gates?)\s*\d+|Ground Floor", en, re.IGNORECASE):
        return False
    if re.match(r"^ATC_(Lorville|Area18|Orison|NewBabbage).*", key, re.IGNORECASE):
        return True
    if re.match(r"^(Lorville|Area18|Orison|NewBabbage)_Destination_.*", key, re.IGNORECASE):
        return not re.fullmatch(r"(?:Floor|City Gates?)\s*\d+|Ground Floor", en, re.IGNORECASE)
    return re.search(
        r"(?:Landing|LandingZone|Spaceport|Station|Hospital|Outpost|Settlement|Racetrack|Gateway|QT)",
        key,
        re.IGNORECASE,
    ) is not None


def derive_term_entries(samples: Iterable[PairSample], max_terms: int = 800) -> tuple[list[TermEntry], dict[str, int]]:
    entries: list[TermEntry] = []
    seen: set[tuple[str, str]] = set()
    conflicts: dict[str, str] = {}
    for sample in samples:
        if sample.source != "global_ini" or sample.category not in {"vehicle", "location"}:
            continue
        if sample.category == "vehicle":
            mineable = any(
                re.match(pattern, sample.key, re.IGNORECASE)
                for pattern in [r"^vehicle_Name.*", r"^(Event|event)_Ship(Name|Title)_.*"]
            )
        else:
            mineable = is_mineable_location_key(sample.key, sample.en)
        if not mineable:
            continue
        if not is_short_term_pair(sample):
            continue
        en = canonical_term_en(sample.en)
        zh = clean_text(sample.zh)
        if en.casefold() in MINED_TERM_DENYLIST:
            continue
        if not en or not zh:
            continue
        if zh in conflicts and conflicts[zh].casefold() != en.casefold():
            continue
        conflicts.setdefault(zh, en)
        key = (zh, en.casefold())
        if key in seen:
            continue
        seen.add(key)
        entries.append(TermEntry(key=f"mined:{sample.key}", category=sample.category, en=en, zh=zh))
        if max_terms > 0 and len(entries) >= max_terms:
            break
    return entries, {"term.mined": len(entries)}


def merge_term_entries(preferred: Iterable[TermEntry], mined: Iterable[TermEntry]) -> list[TermEntry]:
    merged: list[TermEntry] = []
    zh_to_en: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for entry in list(preferred) + list(mined):
        en = canonical_term_en(entry.en)
        zh = clean_text(entry.zh)
        key = (zh, en.casefold())
        if key in seen:
            continue
        current = zh_to_en.get(zh)
        if current and current.casefold() != en.casefold():
            continue
        zh_to_en[zh] = en
        seen.add(key)
        merged.append(TermEntry(key=entry.key, category=entry.category, en=en, zh=zh))
    return merged


def build_term_context_samples(
    entries: Iterable[TermEntry],
    repeat: int = 1,
    max_samples: int = 12000,
) -> tuple[list[PairSample], dict[str, int]]:
    samples: list[PairSample] = []
    vehicle_templates = [
        ("{en} is a ship.", "{zh}是一艘飞船。"),
        ("The {en} is a ship.", "{zh}是一艘船。"),
        ("I want to buy the {en}.", "我想买{zh}。"),
        ("I want to rent the {en}.", "我想租{zh}。"),
        ("Where can I buy the {en}?", "我在哪里可以买{zh}？"),
        ("Where can I rent the {en}?", "我在哪里可以租{zh}？"),
        ("The {en} livery looks good.", "{zh}的涂装很好看。"),
        ("Is the {en} good for beginners?", "{zh}适合新手吗？"),
        ("Is the {en} worth buying?", "{zh}值得买吗？"),
        ("I am flying the {en}.", "我正在驾驶{zh}。"),
        ("I am flying the {en}.", "我开{zh}。"),
        ("I am using the {en} for bounty missions.", "我用{zh}打赏金。"),
        ("I am flying the {en} for bounty missions.", "我开{zh}打赏金。"),
        ("I am doing bounties in the {en}.", "我开{zh}做赏金任务。"),
        ("I am chaining bounty missions in the {en}.", "我开{zh}连刷赏金任务。"),
        ("I am using the {en} for VLRT bounty missions.", "我用{zh}打VLRT赏金。"),
        ("I am using the {en} for MRT bounty missions.", "我用{zh}打MRT赏金。"),
        ("I am using the {en} for HRT bounty missions.", "我用{zh}打HRT赏金。"),
        ("I am using the {en} for VHRT bounty missions.", "我用{zh}打VHRT赏金。"),
        ("I am using the {en} for ERT bounty missions.", "我用{zh}打ERT赏金。"),
        ("Anyone want to join me in the {en}?", "有没有人一起开{zh}？"),
        ("Anyone want to join me for bounties in the {en}?", "有人一起开{zh}打赏金吗？"),
        ("The {en} has enough firepower for bounty missions.", "{zh}打赏金火力够用。"),
        ("The {en} is good for solo bounty hunting.", "{zh}适合单人打赏金。"),
        ("The {en} is not just a literal pirate ship.", "{zh}不是字面意义上的海盗船。"),
        ("My daily ship is the {en}.", "我的日常船是{zh}。"),
        ("The {en} is docked at the station.", "{zh}停靠在空间站。"),
        ("The {en} is about to explode. Run!", "{zh}要爆炸了，快跑！"),
        ("The {en} near the station is about to explode.", "空间站附近的{zh}要爆炸了。"),
        ("The {en} is badly damaged; get away from the hangar.", "{zh}损坏很严重，离机库远一点。"),
        ("The {en} lost shields during the bounty fight.", "{zh}在赏金战斗里掉盾了。"),
        ("Player: The {en} is ready.", "玩家：{zh}准备好了。"),
        ("Player: I am flying the {en} for bounties.", "玩家：我开{zh}打赏金。"),
        (
            "Player: I accepted the bounty contract and will bring the {en}; wait for me at the marker.",
            "玩家：我接了赏金合约，会开{zh}过去，你在标记点等我。",
        ),
        ("Pilot: I need parts for the {en}.", "飞行员：我需要{zh}的配件。"),
        ("New player: Should I choose the {en}?", "新手玩家：我应该选择{zh}吗？"),
        (
            "New player: I only have the {en}; can it handle beginner bounty missions?",
            "新手玩家：我只有{zh}，能不能打入门赏金任务？",
        ),
    ]
    location_templates = [
        ("I am in {en}.", "我在{zh}。"),
        ("I am at {en}.", "我在{zh}。"),
        ("Where is {en}?", "{zh}在哪里？"),
        ("How do I get to {en}?", "我怎么去{zh}？"),
        ("Set a route to {en}.", "设置前往{zh}的路线。"),
        ("Set quantum travel to {en}.", "设置量子航行到{zh}。"),
        ("Can I buy ship weapons in {en}?", "我能在{zh}购买飞船武器吗？"),
        ("Where can I buy ship weapons in {en}?", "我在{zh}哪里可以买到飞船武器？"),
        ("The official location name is {en}.", "{zh}的官方地点名称是{zh}。"),
        ("The map marker says {en}.", "地图标记显示{zh}。"),
        ("The station marker says {en}.", "空间站标记显示{zh}。"),
        ("Where can I land in {en}?", "我在{zh}哪里可以降落？"),
        ("The mission starts in {en}.", "任务从{zh}开始。"),
        ("The ship is docked at {en}.", "飞船停靠在{zh}。"),
        ("There is a ship near {en}.", "{zh}附近有一艘船。"),
        ("Something is about to explode at {en}. Run!", "{zh}有东西要爆炸了，快跑！"),
        ("Set my regeneration point at {en}.", "把我的复活点绑定在{zh}。"),
        ("I accepted a cargo mission from {en}.", "我在{zh}接了货运任务。"),
        ("I accepted a bounty mission near {en}.", "我接了{zh}附近的赏金任务。"),
        ("The hangar at {en} is bugged again.", "{zh}的机库又出问题了。"),
        ("Can I repair and refuel at {en}?", "我能在{zh}维修和补给吗？"),
        ("I need to restock ammunition at {en}.", "我要去{zh}补弹药。"),
        ("Player: Meet me at {en}.", "玩家：在{zh}见我。"),
        ("Player: Meet me at {en}.", "玩家：来{zh}找我。"),
        (
            "Player: I set my spawn at {en}; after this bounty mission I will fly back to repair.",
            "玩家：我把复活点绑在{zh}了，打完这单赏金就飞回去维修。",
        ),
        ("ATC: Welcome to {en}.", "空管：欢迎来到{zh}。"),
    ]
    generic_templates = [
        ("{en}", "{zh}"),
        ("Open {en}.", "打开{zh}。"),
        ("Select {en}.", "选择{zh}。"),
        ("Search for {en}.", "搜索{zh}。"),
        ("I found {en}.", "我找到了{zh}。"),
        ("The marker says {en}.", "标记显示{zh}。"),
    ]
    for entry in entries:
        templates = generic_templates[:]
        if entry.category == "vehicle":
            templates.extend(vehicle_templates)
        elif entry.category == "location":
            templates.extend(location_templates)
            if entry.en.endswith(" Station") and "空间站" in entry.zh:
                templates.extend(
                    [
                        ("{en} is the official station name.", "{zh}的官方英文名是{en}。"),
                        ("Use the official name {en}.", "{zh}要使用官方名称{en}。"),
                        ("Do not expand the station name beyond {en}.", "{zh}不要扩写成非官方站名。"),
                    ]
                )
        for repeat_index in range(max(1, repeat)):
            for template_index, (en_template, zh_template) in enumerate(templates, start=1):
                samples.append(
                    PairSample(
                        key=f"term_context:{entry.key}:{repeat_index + 1}:{template_index}",
                        en=en_template.format(en=entry.en, zh=entry.zh),
                        zh=zh_template.format(en=entry.en, zh=entry.zh),
                        category=entry.category,
                        is_priority=True,
                        source="term_context",
                    )
                )
                if max_samples > 0 and len(samples) >= max_samples:
                    return samples, {"term.context_samples": len(samples)}
    return samples, {"term.context_samples": len(samples)}


def term_priority_key(entry: TermEntry) -> tuple[int, str]:
    return (1 if entry.key.startswith("mined:") else 0, entry.key)


def build_dialogue_context_samples(
    entries: Iterable[TermEntry],
    repeat: int = 1,
    max_samples: int = 20000,
    max_vehicles: int = 180,
    max_locations: int = 80,
) -> tuple[list[PairSample], dict[str, int]]:
    term_entries = sorted(list(entries), key=term_priority_key)
    vehicles = [entry for entry in term_entries if entry.category == "vehicle"][:max_vehicles]
    locations = [entry for entry in term_entries if entry.category == "location"][:max_locations]
    templates = [
        (
            "The {vehicle_en} is docked at {location_en}.",
            "{vehicle_zh}停靠在{location_zh}。",
        ),
        (
            "I saw a {vehicle_en} near {location_en}.",
            "我在{location_zh}附近看到一艘{vehicle_zh}。",
        ),
        (
            "The {vehicle_en} near {location_en} is about to explode. Evacuate!",
            "{location_zh}附近那艘{vehicle_zh}快爆了，赶紧撤离！",
        ),
        (
            "The {vehicle_en} at {location_en} is about to explode. Move away!",
            "{location_zh}的{vehicle_zh}快要爆炸了，离远点！",
        ),
        (
            "The {vehicle_en} at {location_en} is on fire. Run!",
            "{location_zh}的{vehicle_zh}着火了，快跑！",
        ),
        (
            "The {vehicle_en} at {location_en} is going critical. Clear the pad!",
            "{location_zh}的{vehicle_zh}快炸了，离开停机坪！",
        ),
        (
            "The {vehicle_en} at {location_en} is in trouble.",
            "{location_zh}那边的{vehicle_zh}出事了。",
        ),
        (
            "Bounty mission near {location_en}; I am flying the {vehicle_en}.",
            "在{location_zh}附近打赏金，我开{vehicle_zh}。",
        ),
        (
            "Anyone want to run bounties near {location_en} in the {vehicle_en}?",
            "有人一起在{location_zh}附近开{vehicle_zh}打赏金吗？",
        ),
        (
            "Meet at {location_en}; I am bringing the {vehicle_en}.",
            "在{location_zh}集合，我开{vehicle_zh}过去。",
        ),
        (
            "We are taking the {vehicle_en} from {location_en}.",
            "我们从{location_zh}开{vehicle_zh}出发。",
        ),
        (
            "The {vehicle_en} parked outside {location_en} is smoking; do not stand near it.",
            "{location_zh}外面停着的{vehicle_zh}在冒烟，别站太近。",
        ),
        (
            "The {vehicle_en} at {location_en} lost shields during the bounty fight.",
            "{location_zh}那艘{vehicle_zh}打赏金的时候掉盾了。",
        ),
        (
            "After the bounty mission near {location_en}, bring the {vehicle_en} back for repairs.",
            "打完{location_zh}附近的赏金任务后，把{vehicle_zh}开回来修。",
        ),
        (
            "I set my regeneration point at {location_en} and spawned the {vehicle_en} from the terminal.",
            "我把复活点绑在{location_zh}，然后从终端叫出了{vehicle_zh}。",
        ),
        (
            "If the {vehicle_en} explodes near {location_en}, claim it and regroup at the station.",
            "如果{vehicle_zh}在{location_zh}附近炸了，就申领一艘然后在空间站集合。",
        ),
        (
            "The contract marker is near {location_en}; I will take the {vehicle_en} and engage first.",
            "合约标记在{location_zh}附近，我开{vehicle_zh}先上。",
        ),
        (
            "We need one pilot and one gunner for the {vehicle_en} at {location_en}.",
            "{location_zh}这艘{vehicle_zh}还缺一个驾驶和一个炮手。",
        ),
        (
            "Do not translate {vehicle_en} as a generic ship when the Chinese text says {vehicle_zh}.",
            "中文写{vehicle_zh}的时候，不要把它翻成普通飞船。",
        ),
        (
            "Do not translate {location_en} as a generic station when the Chinese text says {location_zh}.",
            "中文写{location_zh}的时候，不要把它翻成普通空间站。",
        ),
        (
            "Player chat: bounty team forming at {location_en}, {vehicle_en} pilot already online.",
            "玩家聊天：{location_zh}组赏金队，{vehicle_zh}驾驶已经上线。",
        ),
        (
            "Player chat: the {vehicle_en} is waiting at {location_en}; bring missiles and medical supplies.",
            "玩家聊天：{vehicle_zh}在{location_zh}等人，带上导弹和医疗补给。",
        ),
        (
            "Player chat: after we repair at {location_en}, we will take the {vehicle_en} for another bounty run.",
            "玩家聊天：我们在{location_zh}修完之后，再开{vehicle_zh}打一轮赏金。",
        ),
        (
            "Long message: I accepted a bounty contract near {location_en}, but the {vehicle_en} is damaged, so wait before quantum travel.",
            "长消息：我接了{location_zh}附近的赏金合约，但是{vehicle_zh}受损了，先别量子跳。",
        ),
        (
            "Long message: if the {vehicle_en} blows up outside {location_en}, move away from the pad and wait for rescue.",
            "长消息：如果{vehicle_zh}在{location_zh}外面的停机坪炸了，先离远点等救援。",
        ),
        (
            "Long message: we will regroup at {location_en}, repair the {vehicle_en}, reload missiles, and continue bounty hunting.",
            "长消息：我们在{location_zh}集合，修好{vehicle_zh}，补完导弹以后继续打赏金。",
        ),
        (
            "Long message: the {vehicle_en} pilot is new, so guide them from {location_en} to the bounty marker.",
            "长消息：{vehicle_zh}驾驶是新人，你从{location_zh}带他去赏金标记点。",
        ),
        (
            "Long message: there are hostiles around {location_en}, and the {vehicle_en} should not undock until escorts arrive.",
            "长消息：{location_zh}周围有敌人，护航到之前{vehicle_zh}先不要离港。",
        ),
    ]

    samples: list[PairSample] = []
    for repeat_index in range(max(1, repeat)):
        for location in locations:
            for vehicle in vehicles:
                for template_index, (en_template, zh_template) in enumerate(templates, start=1):
                    samples.append(
                        PairSample(
                            key=(
                                "term_dialogue:"
                                f"{location.key}:{vehicle.key}:{repeat_index + 1}:{template_index}"
                            ),
                            en=en_template.format(
                                location_en=location.en,
                                location_zh=location.zh,
                                vehicle_en=vehicle.en,
                                vehicle_zh=vehicle.zh,
                            ),
                            zh=zh_template.format(
                                location_en=location.en,
                                location_zh=location.zh,
                                vehicle_en=vehicle.en,
                                vehicle_zh=vehicle.zh,
                            ),
                            category="dialogue",
                            is_priority=True,
                            source="term_dialogue",
                        )
                    )
                    if max_samples > 0 and len(samples) >= max_samples:
                        return samples, {"term.dialogue_samples": len(samples)}
    return samples, {"term.dialogue_samples": len(samples)}


def build_term_contrast_samples(
    entries: Iterable[TermEntry],
    repeat: int = 1,
    max_pairs_per_category: int = 400,
) -> tuple[list[PairSample], dict[str, int]]:
    by_category: dict[str, list[TermEntry]] = {}
    for entry in entries:
        by_category.setdefault(entry.category, []).append(entry)

    samples: list[PairSample] = []
    templates = [
        ("{a_en} is not {b_en}.", "{a_zh}不是{b_zh}。"),
        ("Choose {a_en}, not {b_en}.", "选择{a_zh}，不是{b_zh}。"),
        ("I said {a_en}, not {b_en}.", "我说的是{a_zh}，不是{b_zh}。"),
        ("Compare {a_en} with {b_en}.", "比较{a_zh}和{b_zh}。"),
    ]
    for category, category_entries in sorted(by_category.items()):
        if len(category_entries) < 2:
            continue
        pairs = 0
        for left_index, left in enumerate(category_entries):
            for right in category_entries[left_index + 1 :]:
                if left.zh == right.zh or left.en.casefold() == right.en.casefold():
                    continue
                for repeat_index in range(max(1, repeat)):
                    for template_index, (en_template, zh_template) in enumerate(templates, start=1):
                        samples.append(
                            PairSample(
                                key=(
                                    f"term_contrast:{category}:{left.key}:{right.key}:"
                                    f"{repeat_index + 1}:{template_index}"
                                ),
                                en=en_template.format(
                                    a_en=left.en,
                                    b_en=right.en,
                                    a_zh=left.zh,
                                    b_zh=right.zh,
                                ),
                                zh=zh_template.format(
                                    a_en=left.en,
                                    b_en=right.en,
                                    a_zh=left.zh,
                                    b_zh=right.zh,
                                ),
                                category=category,
                                is_priority=True,
                                source="term_contrast",
                            )
                        )
                pairs += 1
                if pairs >= max_pairs_per_category:
                    break
            if pairs >= max_pairs_per_category:
                break
    return samples, {"term.contrast_samples": len(samples)}


def english_name_present(text: str, english: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(english)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


def has_term_conflict(en: str, zh: str, entries: Iterable[TermEntry]) -> bool:
    term_entries = list(entries)
    for entry in term_entries:
        if entry.zh and entry.zh in zh and not english_name_present(en, entry.en):
            return True
    return False


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


def write_terms_tsv(path: Path, entries: Iterable[TermEntry]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("key\tcategory\ten\tzh\n")
        for entry in entries:
            f.write(
                "\t".join(
                    [
                        entry.key,
                        entry.category,
                        entry.en.replace("\t", " ").replace("\n", " "),
                        entry.zh.replace("\t", " ").replace("\n", " "),
                    ]
                )
                + "\n"
            )
            count += 1
    return count
