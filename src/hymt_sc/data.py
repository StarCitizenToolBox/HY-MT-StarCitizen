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


def load_ship_alias_entries(
    path: Path,
    existing_pairs: Iterable[tuple[str, str]] | None = None,
) -> tuple[list[TermEntry], int]:
    if not path.exists():
        return [], 0
    existing = {(clean_text(zh), clean_text(en).casefold()) for zh, en in (existing_pairs or [])}
    entries: list[TermEntry] = []
    skipped_existing = 0
    seen: set[tuple[str, str]] = set()
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
        dedupe_key = (zh, en.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        entries.append(TermEntry(key=f"ship_alias:{key}", category=category, en=en, zh=zh))
    return entries, skipped_existing


def build_quant_focus_samples(
    term_entries: Iterable[TermEntry],
    alias_entries: Iterable[TermEntry],
    term_repeat: int = 1,
    alias_repeat: int = 1,
) -> tuple[list[PairSample], dict[str, int]]:
    samples: list[PairSample] = []
    templates = [
        ("{en}", "{zh}"),
        ("{en}", "“{zh}”"),
        ("The Star Citizen term is {en}.", "《星际公民》术语是{zh}。"),
        ("The correct English name is {en}.", "正确英文名是{zh}。"),
        ("Use {en}.", "使用{zh}。"),
        ("Do not change {en} into another term.", "不要把{zh}改成其他术语。"),
    ]
    vehicle_templates = [
        ("The ship is {en}.", "这艘船是{zh}。"),
        ("The ship name is {en}.", "船名是{zh}。"),
        ("I said {en}, not another ship.", "我说的是{zh}，不是别的船。"),
        ("Do not translate {en} as a generic ship.", "不要把{zh}翻成普通飞船。"),
        ("I am flying the {en}.", "我开{zh}。"),
        ("I am using the {en} for bounty missions.", "我用{zh}打赏金。"),
        ("The {en} is in the hangar.", "{zh}在机库里。"),
        ("The {en} is ready.", "{zh}准备好了。"),
        ("The {en} is a Star Citizen ship.", "{zh}是《星际公民》里的船。"),
    ]
    location_templates = [
        ("The location is {en}.", "这个地点是{zh}。"),
        ("The station is {en}.", "这个空间站是{zh}。"),
        ("I am at {en}.", "我在{zh}。"),
        ("Meet me at {en}.", "来{zh}找我。"),
        ("Set a route to {en}.", "设置前往{zh}的路线。"),
        ("Do not expand {en} into another location.", "不要把{zh}扩写成其他地点。"),
        ("The marker says {en}.", "标记显示{zh}。"),
        ("The official location name is {en}.", "{zh}是官方地点名称。"),
        ("The official station name is {en}.", "{zh}是官方空间站名称。"),
    ]

    entries = list(term_entries) + list(alias_entries)
    seen_entries: set[tuple[str, str, str]] = set()
    for entry in entries:
        dedupe_key = (entry.category, entry.zh, entry.en.casefold())
        if dedupe_key in seen_entries:
            continue
        seen_entries.add(dedupe_key)
        entry_templates = templates[:]
        if entry.category == "vehicle":
            entry_templates.extend(vehicle_templates)
        elif entry.category == "location":
            entry_templates.extend(location_templates)
        repeats = alias_repeat if entry.key.startswith("ship_alias:") else term_repeat
        for repeat_index in range(max(1, repeats)):
            for template_index, (en_template, zh_template) in enumerate(entry_templates, start=1):
                samples.append(
                    PairSample(
                        key=f"quant_focus:{entry.key}:{repeat_index + 1}:{template_index}",
                        en=en_template.format(en=entry.en, zh=entry.zh),
                        zh=zh_template.format(en=entry.en, zh=entry.zh),
                        category=entry.category,
                        is_priority=True,
                        source="quant_focus",
                    )
                )
    return samples, {"quant_focus.samples": len(samples)}


def build_chat_guard_samples(repeat: int = 1) -> tuple[list[PairSample], dict[str, int]]:
    servers = [
        ("EU server", "欧服"),
        ("US server", "美服"),
        ("Asia server", "亚服"),
        ("Australian server", "澳服"),
        ("test server", "测试服"),
        ("live server", "正式服"),
    ]
    ships = [
        ("Ironclad", "铁甲"),
        ("Corsair", "海盗船"),
        ("Cutter", "小刀"),
        ("Glaive", "长刀"),
        ("Polaris", "北极星"),
        ("Perseus", "英仙座"),
        ("Scorpius", "天蝎座"),
        ("Vulcan", "火神"),
    ]
    ambiguous_ships = [
        ("Corsair", "海盗船", "pirate ship", "海盗船"),
        ("Cutter", "小刀", "knife", "小刀"),
        ("Glaive", "长刀", "long blade", "长刀"),
        ("Polaris", "北极星", "north star", "北极星"),
        ("Perseus", "英仙座", "Perseus constellation", "英仙座"),
        ("Scorpius", "天蝎座", "Scorpius constellation", "天蝎座"),
        ("Vulcan", "火神", "fire god", "火神"),
    ]
    gameplay_terms = [
        ("soft death", "软死亡"),
        ("crime stat", "犯罪等级"),
        ("red", "红名"),
        ("bounty missions", "打赏金"),
        ("bounty target", "赏金目标"),
        ("cargo hauling", "跑货"),
        ("escort", "护航"),
        ("medical beacon", "医疗信标"),
        ("medical rescue", "医疗救援"),
        ("boarding", "登船"),
        ("bunker mission", "地堡任务"),
        ("quantum fuel", "量子燃料"),
        ("claim timer", "申领时间"),
        ("turret seat", "炮塔位"),
        ("missile lock", "锁导弹"),
        ("desync", "同步很差"),
    ]
    locations = [
        ("Seraphim", "炽天使"),
        ("Seraphim Station", "炽天使空间站"),
        ("Lorville", "洛维尔"),
        ("Area18", "18区"),
        ("Orison", "奥里森"),
        ("New Babbage", "新巴贝奇"),
        ("Everus Harbor", "埃弗勒斯港"),
        ("Port Tressler", "特雷斯勒港"),
    ]
    location_spots = [
        ("outside {location_en}", "{location_zh}外面"),
        ("near {location_en}", "{location_zh}附近"),
        ("at the hangar in {location_en}", "{location_zh}机库"),
        ("on the pad at {location_en}", "{location_zh}停机坪"),
        ("by the station entrance at {location_en}", "{location_zh}入口"),
        ("above {location_en}", "{location_zh}上空"),
    ]
    server_templates = [
        ("I am on the {server_en}.", "我在{server_zh}。"),
        ("I switched to the {server_en}.", "我换到{server_zh}了。"),
        ("The {server_en} is lagging today.", "{server_zh}今天很卡。"),
        ("Do not translate {server_en} as a service name.", "不要把{server_zh}翻成服务名称。"),
        ("{server_en} means a game server region.", "{server_zh}指的是游戏服务器区域。"),
        ("People are fighting at the outpost on the {server_en}.", "{server_zh}前哨站有人打架。"),
    ]
    ship_chat_templates = [
        (
            "I got robbed on the {server_en}; my {ship_en} had just been filled with cargo, then I got killed.",
            "我在{server_zh}被打劫了，{ship_zh}刚装满货，就被人打死。",
        ),
        (
            "I was robbed on the {server_en}; the {ship_en} was full of cargo, and then someone killed me.",
            "我在{server_zh}被抢了，{ship_zh}装满了货，然后被人打死了。",
        ),
        (
            "Someone pirated me on the {server_en} after I loaded the {ship_en} with cargo.",
            "我在{server_zh}刚给{ship_zh}装完货就被海盗了。",
        ),
        (
            "I loaded the {ship_en} on the {server_en}, got interdicted, and died before I could escape.",
            "我在{server_zh}给{ship_zh}装货，结果被拦截，没跑掉就死了。",
        ),
        (
            "My {ship_en} was packed with cargo on the {server_en}, and I got killed by another player.",
            "我在{server_zh}的{ship_zh}满载货物，被别的玩家打死了。",
        ),
        (
            "I was doing cargo on the {server_en} in the {ship_en}, but someone killed me near the outpost.",
            "我在{server_zh}开{ship_zh}跑货，在前哨站附近被人打死了。",
        ),
        (
            "The {ship_en} was loaded, but I was killed on the {server_en} before takeoff.",
            "{ship_zh}已经装好货了，但我在{server_zh}起飞前被杀了。",
        ),
        (
            "I got killed on the {server_en}; the {ship_en} was still full of cargo.",
            "我在{server_zh}被杀了，{ship_zh}里还满是货。",
        ),
        (
            "A pirate killed me on the {server_en}, and my {ship_en} was full of cargo.",
            "有个海盗在{server_zh}把我打死了，我的{ship_zh}还装满了货。",
        ),
        (
            "I was hauling cargo on the {server_en} with the {ship_en}, then got robbed and killed.",
            "我在{server_zh}用{ship_zh}运货，然后被抢还被打死。",
        ),
        (
            "I just filled the {ship_en} with cargo on the {server_en}, and then I got shot dead.",
            "我刚在{server_zh}把{ship_zh}装满货，然后就被人打死了。",
        ),
        (
            "The {ship_en} got destroyed on the {server_en}, but I was the one who got killed first.",
            "{ship_zh}在{server_zh}被炸了，但先被打死的是我。",
        ),
    ]
    location_ship_templates = [
        (
            "There is a {ship_en} at {location_en} firing everywhere.",
            "{location_zh}有个{ship_zh}到处开火。",
        ),
        (
            "There is a {ship_en} at {location_en} shooting everywhere.",
            "{location_zh}有个{ship_zh}到处乱射。",
        ),
        (
            "A {ship_en} at {location_en} is firing at everyone.",
            "{location_zh}有个{ship_zh}在打所有人。",
        ),
        (
            "A {ship_en} near {location_en} is shooting at players.",
            "{location_zh}附近有个{ship_zh}在攻击玩家。",
        ),
        (
            "Someone is flying a {ship_en} at {location_en} and opening fire everywhere.",
            "有人在{location_zh}开{ship_zh}到处开火。",
        ),
        (
            "Watch out, a {ship_en} is shooting around {location_en}.",
            "小心，{location_zh}有个{ship_zh}在乱开火。",
        ),
        (
            "The {ship_en} at {location_en} is not a Hornet; it is a {ship_en}.",
            "{location_zh}那个{ship_zh}不是大黄蜂，就是{ship_zh}。",
        ),
        (
            "Do not collapse {location_en} and {ship_en} into a different ship name.",
            "不要把{location_zh}和{ship_zh}合并成另一个船名。",
        ),
        (
            "The {ship_en} at {location_en} is about to explode. Run!",
            "{location_zh}的{ship_zh}要爆炸了，快跑啊。",
        ),
        (
            "A {ship_en} at {location_en} is asking for help.",
            "{location_zh}那艘{ship_zh}在求救。",
        ),
    ]
    player_chat_templates = [
        (
            "There is a {ship_en} {spot_en} firing everywhere.",
            "{spot_zh}有个{ship_zh}到处开火。",
        ),
        (
            "A {ship_en} {spot_en} is shooting at everyone.",
            "{spot_zh}有个{ship_zh}在打所有人。",
        ),
        (
            "A {ship_en} {spot_en} is shooting at players.",
            "{spot_zh}有个{ship_zh}在攻击玩家。",
        ),
        (
            "Watch out, someone is flying a {ship_en} {spot_en} and spraying fire everywhere.",
            "小心，有人在{spot_zh}开{ship_zh}到处乱射。",
        ),
        (
            "Someone in a {ship_en} is camping {spot_en}.",
            "有人开{ship_zh}在{spot_zh}蹲人。",
        ),
        (
            "A {ship_en} is blocking the hangar {spot_en}.",
            "{spot_zh}有个{ship_zh}堵机库。",
        ),
        (
            "A {ship_en} just rammed someone {spot_en}.",
            "{spot_zh}有个{ship_zh}刚撞人。",
        ),
        (
            "A {ship_en} is griefing players {spot_en}.",
            "{spot_zh}有个{ship_zh}在恶意打人。",
        ),
        (
            "The {ship_en} {spot_en} is red; keep away.",
            "{spot_zh}那个{ship_zh}红名了，离远点。",
        ),
        (
            "The {ship_en} {spot_en} is not friendly.",
            "{spot_zh}那个{ship_zh}不是友军。",
        ),
        (
            "Can anyone help kill the {ship_en} {spot_en}?",
            "有没有人来帮忙打掉{spot_zh}那个{ship_zh}？",
        ),
        (
            "Do not leave the hangar; a {ship_en} is shooting outside {location_en}.",
            "先别出机库，{location_zh}外面有个{ship_zh}在开火。",
        ),
        (
            "I saw a {ship_en} {spot_en}, but I do not know if it is friendly.",
            "我在{spot_zh}看到一个{ship_zh}，不知道是不是友军。",
        ),
        (
            "The {ship_en} {spot_en} destroyed my ship.",
            "{spot_zh}那个{ship_zh}把我的船打爆了。",
        ),
        (
            "The {ship_en} {spot_en} killed me before I could take off.",
            "{spot_zh}那个{ship_zh}在我起飞前把我打死了。",
        ),
        (
            "I was about to land at {location_en}, but a {ship_en} started firing at me.",
            "我准备降落{location_zh}，结果有个{ship_zh}开始打我。",
        ),
        (
            "I was leaving {location_en} when a {ship_en} interdicted me.",
            "我刚离开{location_zh}就被一个{ship_zh}拦截了。",
        ),
        (
            "I am hauling cargo in the {ship_en} near {location_en}.",
            "我在{location_zh}附近开{ship_zh}跑货。",
        ),
        (
            "The {ship_en} is full of cargo and waiting at {location_en}.",
            "{ship_zh}装满货了，在{location_zh}等人。",
        ),
        (
            "Meet at {location_en}; I will bring the {ship_en}.",
            "{location_zh}集合，我开{ship_zh}过去。",
        ),
        (
            "Need one gunner for the {ship_en} at {location_en}.",
            "{location_zh}的{ship_zh}缺一个炮手。",
        ),
        (
            "Need escort for a {ship_en} cargo run from {location_en}.",
            "需要护航，从{location_zh}开{ship_zh}跑货。",
        ),
        (
            "I am repairing the {ship_en} at {location_en}.",
            "我在{location_zh}修{ship_zh}。",
        ),
        (
            "I am refueling the {ship_en} at {location_en}.",
            "我在{location_zh}给{ship_zh}补油。",
        ),
        (
            "The {ship_en} needs ammunition at {location_en}.",
            "{ship_zh}在{location_zh}需要补弹药。",
        ),
        (
            "I accepted a bounty near {location_en} and will bring the {ship_en}.",
            "我接了{location_zh}附近的赏金，会开{ship_zh}过去。",
        ),
        (
            "Anyone want to run bounty missions from {location_en} in a {ship_en}?",
            "有人从{location_zh}开{ship_zh}一起打赏金吗？",
        ),
        (
            "The {ship_en} at {location_en} is damaged; wait for repairs.",
            "{location_zh}那艘{ship_zh}受损了，等修好。",
        ),
        (
            "The {ship_en} at {location_en} is about to explode; move away.",
            "{location_zh}那艘{ship_zh}快炸了，离远点。",
        ),
        (
            "I crashed the {ship_en} near {location_en}; can someone pick me up?",
            "我把{ship_zh}坠在{location_zh}附近了，有人能接我吗？",
        ),
    ]
    server_location_templates = [
        (
            "On the {server_en}, there is a {ship_en} at {location_en} firing everywhere.",
            "{server_zh}{location_zh}有个{ship_zh}到处开火。",
        ),
        (
            "On the {server_en}, a {ship_en} is camping the hangar at {location_en}.",
            "{server_zh}{location_zh}有个{ship_zh}堵机库。",
        ),
        (
            "On the {server_en}, I got killed by a {ship_en} near {location_en}.",
            "我在{server_zh}{location_zh}附近被一个{ship_zh}打死了。",
        ),
        (
            "On the {server_en}, we are meeting at {location_en} and taking the {ship_en}.",
            "我们在{server_zh}{location_zh}集合，开{ship_zh}出发。",
        ),
        (
            "On the {server_en}, I am hauling cargo in the {ship_en} from {location_en}.",
            "我在{server_zh}从{location_zh}开{ship_zh}跑货。",
        ),
        (
            "On the {server_en}, the {ship_en} at {location_en} is asking for escort.",
            "{server_zh}{location_zh}那艘{ship_zh}在喊护航。",
        ),
    ]
    ship_identity_templates = [
        (
            "In Star Citizen chat, this term means the ship {ship_en}, not {literal_en}.",
            "在星际公民聊天里，{ship_zh}是{ship_en}这艘船，不是普通说法里的{literal_zh}。",
        ),
        (
            "Here this is the ship name {ship_en}.",
            "这里的{ship_zh}是船名：{ship_en}。",
        ),
        (
            "When players use this term, translate it as {ship_en}.",
            "玩家说{ship_zh}的时候，要翻成{ship_en}。",
        ),
        (
            "Do not translate the term literally in this game context; use {ship_en}.",
            "这个游戏语境里不要把{ship_zh}直译，应该用{ship_en}。",
        ),
    ]
    gameplay_identity_templates = [
        (
            "In Star Citizen chat, this phrase means {term_en}.",
            "在星际公民聊天里，{term_zh}要翻成{term_en}。",
        ),
        (
            "Here this is gameplay slang: {term_en}.",
            "这里的{term_zh}是游戏黑话：{term_en}。",
        ),
        (
            "When players use this phrase, translate it as {term_en}.",
            "玩家说{term_zh}的时候，要翻成{term_en}。",
        ),
        (
            "Do not translate this phrase as ordinary Chinese; use {term_en}.",
            "不要把{term_zh}按普通中文翻译，要用{term_en}。",
        ),
    ]
    ambiguous_ship_chat_templates = [
        (
            "There is a {ship_en} at {location_en} firing everywhere. Come destroy it.",
            "{location_zh}有个{ship_zh}到处开火，快来打掉。",
        ),
        (
            "A {ship_en} is shooting around {location_en}; do not leave the hangar.",
            "{location_zh}有个{ship_zh}到处乱射，先别出机库。",
        ),
        (
            "The {ship_en} at {location_en} is about to explode. Run!",
            "{location_zh}的{ship_zh}要爆炸了，快跑啊。",
        ),
        (
            "The {ship_en} at {location_en} is red and firing on everyone.",
            "{location_zh}的{ship_zh}红名了，在打所有人。",
        ),
        (
            "Someone is flying a {ship_en} near {location_en} and killing players.",
            "有人在{location_zh}附近开{ship_zh}杀玩家。",
        ),
        (
            "Watch out for the {ship_en} above {location_en}; it is not friendly.",
            "小心{location_zh}上空那艘{ship_zh}，不是友军。",
        ),
        (
            "Need help at {location_en}; a {ship_en} is camping the station.",
            "{location_zh}需要支援，有个{ship_zh}在蹲空间站。",
        ),
        (
            "The {ship_en} near {location_en} destroyed my ship.",
            "{location_zh}附近那艘{ship_zh}把我的船打爆了。",
        ),
        (
            "I accepted a bounty near {location_en} and will bring the {ship_en}.",
            "我接了{location_zh}附近的赏金，开{ship_zh}过去。",
        ),
        (
            "Anyone want to run bounty missions with my {ship_en} from {location_en}?",
            "我开{ship_zh}从{location_zh}打赏金，有没有一起的？",
        ),
        (
            "I am hauling cargo in the {ship_en} from {location_en}; need escort.",
            "我从{location_zh}开{ship_zh}跑货，需要护航。",
        ),
        (
            "The {ship_en} is full of cargo at {location_en}; do not shoot it.",
            "{ship_zh}在{location_zh}满载货物，别打它。",
        ),
        (
            "I crashed the {ship_en} near {location_en}; can anyone pick me up?",
            "我把{ship_zh}坠在{location_zh}附近了，有人能接我吗？",
        ),
        (
            "The target is a {ship_en}, not a Hornet Ghost.",
            "目标是{ship_zh}，不是大黄蜂幽灵。",
        ),
        (
            "I said {ship_en}, not Hornet Ghost.",
            "我说的是{ship_zh}，不是大黄蜂幽灵。",
        ),
    ]
    operation_chat_templates = [
        (
            "I am taking the {ship_en} from {location_en} for bounty missions. Anyone want to join?",
            "我从{location_zh}开{ship_zh}打赏金，有没有一起的？",
        ),
        (
            "I am flying the {ship_en} out of {location_en}; need a gunner and a turret operator.",
            "我从{location_zh}开{ship_zh}出发，缺炮手和炮塔位。",
        ),
        (
            "The {ship_en} at {location_en} needs a pilot; I can take the gunner seat.",
            "{location_zh}那艘{ship_zh}缺驾驶，我可以坐炮手位。",
        ),
        (
            "Bring the {ship_en} to {location_en}; we are forming a party there.",
            "把{ship_zh}开到{location_zh}，我们在那里组队。",
        ),
        (
            "Do not shoot the {ship_en} at {location_en}; it is with our party.",
            "别打{location_zh}那艘{ship_zh}，那是我们队里的。",
        ),
        (
            "The {ship_en} near {location_en} is our escort, not the target.",
            "{location_zh}附近那艘{ship_zh}是护航，不是目标。",
        ),
        (
            "The target is near {location_en}; I will bring the {ship_en} and pull aggro.",
            "目标在{location_zh}附近，我开{ship_zh}过去拉仇恨。",
        ),
        (
            "A hostile {ship_en} is locking missiles near {location_en}. Break away.",
            "{location_zh}附近有敌对{ship_zh}在锁导弹，赶紧脱离。",
        ),
        (
            "The {ship_en} at {location_en} is disabled but not destroyed.",
            "{location_zh}那艘{ship_zh}瘫痪了，但还没炸。",
        ),
        (
            "The {ship_en} near {location_en} is in soft death; board it carefully.",
            "{location_zh}附近那艘{ship_zh}软死亡了，小心登船。",
        ),
        (
            "Someone stole my {ship_en} at {location_en}; mark it as hostile.",
            "有人在{location_zh}偷了我的{ship_zh}，把它标成敌对。",
        ),
        (
            "I cannot open the hangar at {location_en}; the {ship_en} is stuck inside.",
            "{location_zh}机库打不开，{ship_zh}卡在里面了。",
        ),
        (
            "The {ship_en} is stuck on the pad at {location_en}; I need a claim timer.",
            "{ship_zh}卡在{location_zh}停机坪上了，我要等申领时间。",
        ),
        (
            "The {ship_en} lost shields near {location_en}; do not start the jump yet.",
            "{ship_zh}在{location_zh}附近掉盾了，先别跳跃。",
        ),
        (
            "The {ship_en} is out of quantum fuel at {location_en}; can anyone refuel it?",
            "{ship_zh}在{location_zh}没量子燃料了，有人能补油吗？",
        ),
        (
            "The {ship_en} at {location_en} needs repairs before we take another contract.",
            "{location_zh}那艘{ship_zh}要先修好，再接下一个合约。",
        ),
        (
            "The {ship_en} at {location_en} has no ammunition left.",
            "{location_zh}那艘{ship_zh}没弹药了。",
        ),
        (
            "I am loading cargo into the {ship_en} at {location_en}; watch the ramp.",
            "我在{location_zh}给{ship_zh}装货，看一下舱门。",
        ),
        (
            "The cargo in the {ship_en} at {location_en} is valuable; do not leave it unattended.",
            "{location_zh}那艘{ship_zh}里的货很贵，别没人看着。",
        ),
        (
            "The {ship_en} at {location_en} is carrying salvage boxes.",
            "{location_zh}那艘{ship_zh}装的是打捞箱。",
        ),
        (
            "I am using the {ship_en} near {location_en} to scout for salvage.",
            "我在{location_zh}附近开{ship_zh}找打捞目标。",
        ),
        (
            "I found a mining spot near {location_en}; bring the {ship_en} as escort.",
            "我在{location_zh}附近找到矿点了，开{ship_zh}来护航。",
        ),
        (
            "The {ship_en} is waiting at {location_en} while we finish the bunker mission.",
            "我们打完地堡任务前，{ship_zh}先在{location_zh}等着。",
        ),
        (
            "I died near {location_en}; can the {ship_en} pick me up?",
            "我死在{location_zh}附近了，{ship_zh}能来接我吗？",
        ),
        (
            "Put a medical beacon near {location_en}; the {ship_en} can land there.",
            "在{location_zh}附近发医疗信标，{ship_zh}能在那里降落。",
        ),
        (
            "The {ship_en} at {location_en} has a crime stat target on board.",
            "{location_zh}那艘{ship_zh}上有个犯罪等级目标。",
        ),
        (
            "The {ship_en} at {location_en} is not an NPC; it is a player ship.",
            "{location_zh}那艘{ship_zh}不是 NPC，是玩家船。",
        ),
        (
            "Do not accept the party invite until the {ship_en} leaves {location_en}.",
            "{ship_zh}离开{location_zh}之前，先别接受组队邀请。",
        ),
        (
            "The {ship_en} at {location_en} is bait; their friends are hiding nearby.",
            "{location_zh}那艘{ship_zh}是诱饵，他们队友躲在附近。",
        ),
        (
            "I am scanning the {ship_en} at {location_en}; wait before opening fire.",
            "我在扫描{location_zh}那艘{ship_zh}，先别开火。",
        ),
        (
            "The {ship_en} is landed at {location_en}; meet at the rear ramp.",
            "{ship_zh}已经停在{location_zh}了，后舱门集合。",
        ),
        (
            "The {ship_en} is hovering above {location_en}; look up before taking off.",
            "{ship_zh}在{location_zh}上空悬停，起飞前先看头顶。",
        ),
        (
            "I see two ships at {location_en}: one {ship_en} and one hostile escort.",
            "我在{location_zh}看到两艘船：一艘{ship_zh}和一艘敌对护航。",
        ),
        (
            "The {ship_en} at {location_en} keeps circling the station.",
            "{location_zh}那艘{ship_zh}一直绕着空间站飞。",
        ),
        (
            "The {ship_en} jumped away from {location_en}; check the next marker.",
            "{ship_zh}从{location_zh}跳走了，检查下一个标记。",
        ),
        (
            "The {ship_en} near {location_en} is asking for a tow.",
            "{location_zh}附近那艘{ship_zh}在喊拖船。",
        ),
        (
            "The {ship_en} at {location_en} is full; take another ship.",
            "{location_zh}那艘{ship_zh}满员了，换一艘船。",
        ),
        (
            "I will stay in the {ship_en} at {location_en} and watch for boarders.",
            "我留在{location_zh}的{ship_zh}里看登船的人。",
        ),
        (
            "The {ship_en} at {location_en} is safe for now, but keep shields up.",
            "{location_zh}那艘{ship_zh}暂时安全，但盾别关。",
        ),
        (
            "If the {ship_en} at {location_en} turns red, jump out immediately.",
            "{location_zh}那艘{ship_zh}如果变红，马上跳走。",
        ),
    ]
    server_ship_operation_templates = [
        (
            "On the {server_en}, I am flying the {ship_en} for bounty missions. Need one more.",
            "我在{server_zh}开{ship_zh}打赏金，还缺一个人。",
        ),
        (
            "On the {server_en}, the {ship_en} is ready for cargo hauling.",
            "{server_zh}的{ship_zh}已经准备好跑货了。",
        ),
        (
            "On the {server_en}, my {ship_en} is full of cargo and needs escort.",
            "我在{server_zh}的{ship_zh}满货了，需要护航。",
        ),
        (
            "The {ship_en} on the {server_en} has a free turret seat.",
            "{server_zh}的{ship_zh}还有一个炮塔位。",
        ),
        (
            "The {ship_en} on the {server_en} lost shields after the last fight.",
            "{server_zh}的{ship_zh}上一场打完掉盾了。",
        ),
        (
            "On the {server_en}, someone stole a {ship_en}; check before boarding.",
            "{server_zh}有人偷了一艘{ship_zh}，登船前确认一下。",
        ),
        (
            "On the {server_en}, do not shoot the {ship_en}; it is friendly.",
            "{server_zh}别打{ship_zh}，那是友军。",
        ),
        (
            "On the {server_en}, a hostile {ship_en} is hunting cargo runners.",
            "{server_zh}有敌对{ship_zh}在抓跑货的人。",
        ),
        (
            "On the {server_en}, the {ship_en} keeps showing as red even after party invite.",
            "{server_zh}的{ship_zh}组队后还是显示红名。",
        ),
        (
            "On the {server_en}, I claimed the {ship_en}; wait for the timer.",
            "我在{server_zh}申领了{ship_zh}，等一下计时器。",
        ),
        (
            "On the {server_en}, the {ship_en} despawned with all the cargo inside.",
            "{server_zh}的{ship_zh}连货一起消失了。",
        ),
        (
            "On the {server_en}, the {ship_en} is bugged and cannot take off.",
            "{server_zh}的{ship_zh}出 bug 了，起飞不了。",
        ),
        (
            "On the {server_en}, I need someone to crew the {ship_en}.",
            "我在{server_zh}需要人来上{ship_zh}当船员。",
        ),
        (
            "On the {server_en}, we can use the {ship_en} as bait.",
            "{server_zh}我们可以拿{ship_zh}当诱饵。",
        ),
        (
            "On the {server_en}, the {ship_en} is too damaged for another fight.",
            "{server_zh}的{ship_zh}损伤太重，不能再打下一场。",
        ),
        (
            "On the {server_en}, bring the {ship_en} if you want to join the fleet.",
            "{server_zh}想进舰队就把{ship_zh}开过来。",
        ),
        (
            "On the {server_en}, I am testing whether the {ship_en} still has cargo after recovery.",
            "我在{server_zh}测试{ship_zh}找回后货还在不在。",
        ),
        (
            "On the {server_en}, the {ship_en} is the ship we are talking about.",
            "{server_zh}我们说的就是{ship_zh}这艘船。",
        ),
        (
            "On the {server_en}, do not translate this ship name as a normal word; it is {ship_en}.",
            "{server_zh}这里别把{ship_zh}当普通词，它是{ship_en}。",
        ),
        (
            "On the {server_en}, I said {ship_en}, not a different ship.",
            "{server_zh}我说的是{ship_zh}，不是别的船。",
        ),
    ]
    location_status_templates = [
        (
            "On the {server_en}, the elevators at {location_en} are broken.",
            "{server_zh}{location_zh}的电梯坏了。",
        ),
        (
            "On the {server_en}, {location_en} is full of hostile players.",
            "{server_zh}{location_zh}全是敌对玩家。",
        ),
        (
            "On the {server_en}, meet at {location_en} and do not open fire.",
            "{server_zh}{location_zh}集合，先别开火。",
        ),
        (
            "On the {server_en}, {location_en} is safe for landing right now.",
            "{server_zh}{location_zh}现在可以安全降落。",
        ),
        (
            "On the {server_en}, do not land at {location_en}; someone is camping the pads.",
            "{server_zh}别降落{location_zh}，有人蹲停机坪。",
        ),
        (
            "On the {server_en}, the hangars at {location_en} are not opening.",
            "{server_zh}{location_zh}的机库门打不开。",
        ),
        (
            "On the {server_en}, my cargo disappeared at {location_en}.",
            "我在{server_zh}{location_zh}货物消失了。",
        ),
        (
            "On the {server_en}, I died at {location_en}; can someone revive me?",
            "我在{server_zh}{location_zh}死了，有人能救我吗？",
        ),
        (
            "On the {server_en}, medical rescue is needed near {location_en}.",
            "{server_zh}{location_zh}附近需要医疗救援。",
        ),
        (
            "On the {server_en}, a bounty target is hiding at {location_en}.",
            "{server_zh}有个赏金目标躲在{location_zh}。",
        ),
        (
            "On the {server_en}, security at {location_en} is shooting everyone.",
            "{server_zh}{location_zh}的安保在打所有人。",
        ),
        (
            "On the {server_en}, I got a crime stat near {location_en}.",
            "我在{server_zh}{location_zh}附近红名了。",
        ),
        (
            "On the {server_en}, clear your crime stat before coming to {location_en}.",
            "{server_zh}来{location_zh}之前先清犯罪等级。",
        ),
        (
            "On the {server_en}, {location_en} has heavy desync.",
            "{server_zh}{location_zh}同步很差。",
        ),
        (
            "On the {server_en}, {location_en} is lagging but still playable.",
            "{server_zh}{location_zh}很卡，但还能玩。",
        ),
        (
            "On the {server_en}, the shop terminals at {location_en} are not working.",
            "{server_zh}{location_zh}的商店终端用不了。",
        ),
        (
            "On the {server_en}, there are pirates waiting outside {location_en}.",
            "{server_zh}{location_zh}外面有海盗在等人。",
        ),
        (
            "On the {server_en}, do not bring cargo to {location_en} yet.",
            "{server_zh}先别把货带到{location_zh}。",
        ),
        (
            "On the {server_en}, I need pickup near {location_en}.",
            "我在{server_zh}{location_zh}附近需要接送。",
        ),
        (
            "On the {server_en}, the landing marker at {location_en} is wrong.",
            "{server_zh}{location_zh}的降落标记不对。",
        ),
        (
            "On the {server_en}, {location_en} is where the party is regrouping.",
            "{server_zh}{location_zh}是队伍重新集合的地方。",
        ),
        (
            "On the {server_en}, someone is asking for escort at {location_en}.",
            "{server_zh}{location_zh}有人在喊护航。",
        ),
        (
            "On the {server_en}, I can sell cargo at {location_en}.",
            "我可以在{server_zh}{location_zh}卖货。",
        ),
        (
            "On the {server_en}, {location_en} is not the target; it is the meetup point.",
            "{server_zh}{location_zh}不是目标，是集合点。",
        ),
        (
            "On the {server_en}, wait at {location_en} until everyone joins voice.",
            "{server_zh}在{location_zh}等大家进语音。",
        ),
    ]
    gameplay_jargon_contexts = [
        (
            "The {ship_en} near {location_en} is in soft death; board carefully.",
            "{location_zh}附近那艘{ship_zh}软死亡了，小心登船。",
        ),
        (
            "The {ship_en} at {location_en} is in soft death, not fully destroyed.",
            "{location_zh}那艘{ship_zh}是软死亡，不是彻底炸了。",
        ),
        (
            "The {ship_en} at {location_en} is red; do not stand near it.",
            "{location_zh}那艘{ship_zh}红名了，别站太近。",
        ),
        (
            "There is a crime stat target inside the {ship_en} at {location_en}.",
            "{location_zh}那艘{ship_zh}里面有犯罪等级目标。",
        ),
        (
            "I am doing bounty missions in the {ship_en} near {location_en}.",
            "我在{location_zh}附近开{ship_zh}打赏金。",
        ),
        (
            "I found the bounty target near {location_en}; bring the {ship_en}.",
            "我在{location_zh}附近找到赏金目标了，把{ship_zh}开过来。",
        ),
        (
            "I am cargo hauling from {location_en} in the {ship_en}; escort me.",
            "我从{location_zh}开{ship_zh}跑货，来护航我。",
        ),
        (
            "Set a medical beacon near {location_en}; the {ship_en} can pick you up.",
            "在{location_zh}附近发医疗信标，{ship_zh}可以接你。",
        ),
        (
            "Medical rescue is needed near {location_en}; the {ship_en} can land there.",
            "{location_zh}附近需要医疗救援，{ship_zh}可以在那里降落。",
        ),
        (
            "We are boarding the {ship_en} at {location_en}; hold fire.",
            "我们要登船{location_zh}那艘{ship_zh}，先别开火。",
        ),
        (
            "We are doing a bunker mission near {location_en}; leave the {ship_en} outside.",
            "我们在{location_zh}附近做地堡任务，{ship_zh}停外面。",
        ),
        (
            "The {ship_en} at {location_en} is out of quantum fuel.",
            "{location_zh}那艘{ship_zh}没量子燃料了。",
        ),
        (
            "The {ship_en} claim timer is almost done at {location_en}.",
            "{ship_zh}在{location_zh}的申领时间快结束了。",
        ),
        (
            "The {ship_en} at {location_en} still has a free turret seat.",
            "{location_zh}那艘{ship_zh}还有一个炮塔位。",
        ),
        (
            "A hostile {ship_en} near {location_en} has missile lock on me.",
            "{location_zh}附近有敌对{ship_zh}在锁我导弹。",
        ),
        (
            "{location_en} has heavy desync, so the {ship_en} may rubber-band.",
            "{location_zh}同步很差，{ship_zh}可能会来回瞬移。",
        ),
    ]

    samples: list[PairSample] = []
    for repeat_index in range(max(1, repeat)):
        for ship_index, (ship_en, ship_zh, literal_en, literal_zh) in enumerate(ambiguous_ships, start=1):
            for template_index, (en_template, zh_template) in enumerate(ship_identity_templates, start=1):
                samples.append(
                    PairSample(
                        key=f"chat_guard:ship_identity:{ship_index}:{repeat_index + 1}:{template_index}",
                        en=en_template.format(
                            ship_en=ship_en,
                            ship_zh=ship_zh,
                            literal_en=literal_en,
                            literal_zh=literal_zh,
                        ),
                        zh=zh_template.format(ship_en=ship_en, ship_zh=ship_zh, literal_en=literal_en, literal_zh=literal_zh),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
        for term_index, (term_en, term_zh) in enumerate(gameplay_terms, start=1):
            for template_index, (en_template, zh_template) in enumerate(gameplay_identity_templates, start=1):
                samples.append(
                    PairSample(
                        key=f"chat_guard:gameplay_identity:{term_index}:{repeat_index + 1}:{template_index}",
                        en=en_template.format(term_en=term_en, term_zh=term_zh),
                        zh=zh_template.format(term_en=term_en, term_zh=term_zh),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
        for server_index, (server_en, server_zh) in enumerate(servers, start=1):
            for template_index, (en_template, zh_template) in enumerate(server_templates, start=1):
                samples.append(
                    PairSample(
                        key=f"chat_guard:server:{server_index}:{repeat_index + 1}:{template_index}",
                        en=en_template.format(server_en=server_en, server_zh=server_zh),
                        zh=zh_template.format(server_en=server_en, server_zh=server_zh),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
            for ship_index, (ship_en, ship_zh) in enumerate(ships, start=1):
                for template_index, (en_template, zh_template) in enumerate(ship_chat_templates, start=1):
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:cargo:{server_index}:{ship_index}:"
                                f"{repeat_index + 1}:{template_index}"
                            ),
                            en=en_template.format(
                                server_en=server_en,
                                server_zh=server_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            ),
                            zh=zh_template.format(
                                server_en=server_en,
                                server_zh=server_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for template_index, (en_template, zh_template) in enumerate(
                    server_ship_operation_templates,
                    start=1,
                ):
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:server_ship_operation:{server_index}:{ship_index}:"
                                f"{repeat_index + 1}:{template_index}"
                            ),
                            en=en_template.format(
                                server_en=server_en,
                                server_zh=server_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            ),
                            zh=zh_template.format(
                                server_en=server_en,
                                server_zh=server_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
        for location_index, (location_en, location_zh) in enumerate(locations, start=1):
            for ship_index, (ship_en, ship_zh) in enumerate(ships, start=1):
                for template_index, (en_template, zh_template) in enumerate(location_ship_templates, start=1):
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:location_fire:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{template_index}"
                            ),
                            en=en_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            ),
                            zh=zh_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for template_index, (en_template, zh_template) in enumerate(operation_chat_templates, start=1):
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:operation:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{template_index}"
                            ),
                            en=en_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            ),
                            zh=zh_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for template_index, (en_template, zh_template) in enumerate(gameplay_jargon_contexts, start=1):
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:gameplay_context:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{template_index}"
                            ),
                            en=en_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            ),
                            zh=zh_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for spot_index, (spot_en_template, spot_zh_template) in enumerate(location_spots, start=1):
                    spot_en = spot_en_template.format(location_en=location_en, location_zh=location_zh)
                    spot_zh = spot_zh_template.format(location_en=location_en, location_zh=location_zh)
                    for template_index, (en_template, zh_template) in enumerate(player_chat_templates, start=1):
                        samples.append(
                            PairSample(
                                key=(
                                    f"chat_guard:player:{location_index}:{spot_index}:{ship_index}:"
                                    f"{repeat_index + 1}:{template_index}"
                                ),
                                en=en_template.format(
                                    location_en=location_en,
                                    location_zh=location_zh,
                                    spot_en=spot_en,
                                    spot_zh=spot_zh,
                                    ship_en=ship_en,
                                    ship_zh=ship_zh,
                                ),
                                zh=zh_template.format(
                                    location_en=location_en,
                                    location_zh=location_zh,
                                    spot_en=spot_en,
                                    spot_zh=spot_zh,
                                    ship_en=ship_en,
                                    ship_zh=ship_zh,
                                ),
                                category="chat",
                                is_priority=True,
                                source="chat_guard",
                            )
                        )
            for ship_index, (ship_en, ship_zh, literal_en, _literal_zh) in enumerate(ambiguous_ships, start=1):
                for template_index, (en_template, zh_template) in enumerate(ambiguous_ship_chat_templates, start=1):
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:ambiguous_ship:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{template_index}"
                            ),
                            en=en_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                                literal_en=literal_en,
                            ),
                            zh=zh_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                                literal_en=literal_en,
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
        for server_index, (server_en, server_zh) in enumerate(servers, start=1):
            for location_index, (location_en, location_zh) in enumerate(locations, start=1):
                for template_index, (en_template, zh_template) in enumerate(location_status_templates, start=1):
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:location_status:{server_index}:{location_index}:"
                                f"{repeat_index + 1}:{template_index}"
                            ),
                            en=en_template.format(
                                server_en=server_en,
                                server_zh=server_zh,
                                location_en=location_en,
                                location_zh=location_zh,
                            ),
                            zh=zh_template.format(
                                server_en=server_en,
                                server_zh=server_zh,
                                location_en=location_en,
                                location_zh=location_zh,
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for ship_index, (ship_en, ship_zh) in enumerate(ships, start=1):
                    for template_index, (en_template, zh_template) in enumerate(server_location_templates, start=1):
                        samples.append(
                            PairSample(
                                key=(
                                    f"chat_guard:server_location:{server_index}:{location_index}:"
                                    f"{ship_index}:{repeat_index + 1}:{template_index}"
                                ),
                                en=en_template.format(
                                    server_en=server_en,
                                    server_zh=server_zh,
                                    location_en=location_en,
                                    location_zh=location_zh,
                                    ship_en=ship_en,
                                    ship_zh=ship_zh,
                                ),
                                zh=zh_template.format(
                                    server_en=server_en,
                                    server_zh=server_zh,
                                    location_en=location_en,
                                    location_zh=location_zh,
                                    ship_en=ship_en,
                                    ship_zh=ship_zh,
                                ),
                                category="chat",
                                is_priority=True,
                                source="chat_guard",
                            )
                        )
    return samples, {"chat_guard.samples": len(samples)}


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
