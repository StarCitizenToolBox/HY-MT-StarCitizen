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
        if contains_cjk(en):
            skipped_existing += 1
            continue
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
        if contains_cjk(en):
            skipped_existing += 1
            continue
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
    alias_chat_locations = [
        ("Seraphim", "炽天使"),
        ("Seraphim Station", "炽天使空间站"),
        ("Lorville", "洛维尔"),
        ("Area18", "18区"),
        ("Orison", "奥里森"),
        ("New Babbage", "新巴贝奇"),
        ("Everus Harbor", "埃弗勒斯港"),
        ("Port Tressler", "特雷斯勒港"),
    ]
    alias_chat_templates = [
        ("There is a {en} at {location_en} firing everywhere.", "{location_zh}有个{zh}到处开火。"),
        ("The {en} at {location_en} is firing everywhere.", "{location_zh}有个{zh}到处开火。"),
        ("The {en} near {location_en} is shooting at players.", "{location_zh}附近有个{zh}在攻击玩家。"),
        ("I am taking the {en} from {location_en} for bounty missions.", "我从{location_zh}开{zh}打赏金。"),
        ("Anyone want to run bounty missions in the {en} from {location_en}?", "有人从{location_zh}开{zh}一起打赏金吗？"),
        ("I am hauling cargo in the {en} from {location_en}; need escort.", "我从{location_zh}开{zh}跑货，需要护航。"),
        ("The {en} at {location_en} is in soft death; board carefully.", "{location_zh}那艘{zh}软死亡了，小心登船。"),
        ("The {en} near {location_en} is locking missiles.", "{location_zh}附近那艘{zh}在锁导弹。"),
        ("The {en} at {location_en} needs quantum fuel before we jump.", "{location_zh}那艘{zh}跳跃前需要量子燃料。"),
        ("I dropped a medical beacon near {location_en}; the {en} can land there.", "我在{location_zh}附近发了医疗信标，{zh}能在那里降落。"),
        ("Medical rescue is needed near {location_en}; bring the {en}.", "{location_zh}附近需要医疗救援，把{zh}开过来。"),
        ("We are doing a bunker mission near {location_en}; leave the {en} outside.", "我们在{location_zh}附近做地堡任务，{zh}停外面。"),
        ("The {en} claim timer at {location_en} is almost done.", "{zh}在{location_zh}的申领时间快好了。"),
        ("{location_en} has heavy desync, so the {en} may rubber-band.", "{location_zh}同步很差，{zh}可能会来回瞬移。"),
        ("The {en} at {location_en} needs a gunner and a turret seat.", "{location_zh}那艘{zh}缺炮手和炮塔位。"),
        ("The {en} at {location_en} has a crime stat target on board.", "{location_zh}那艘{zh}上有犯罪等级目标。"),
        ("The bounty target near {location_en} is flying the {en}.", "{location_zh}附近的赏金目标开着{zh}。"),
        ("The {en} at {location_en} is red; do not stand near it.", "{location_zh}那艘{zh}红名了，别站太近。"),
        ("Do not confuse this ship: it is the {en}.", "别把这艘船认错，它是{zh}。"),
        ("Global chat: the {en} at {location_en} is not friendly.", "全局频道：{location_zh}那艘{zh}不是友军。"),
        ("Voice chat: bring the {en} to {location_en}.", "语音里说：把{zh}开到{location_zh}。"),
        ("Warning, the {en} is camping the hangar at {location_en}.", "注意，{location_zh}有个{zh}在蹲机库。"),
        ("The target is a {en}, not a Hornet Ghost.", "目标是{zh}，不是大黄蜂幽灵。"),
        ("The {en} at {location_en} is firing everywhere. > F7C-S Hornet Ghost", "{location_zh}有个{zh}到处开火。 >F7C-S Hornet Ghost"),
    ]
    alias_chat_noise_templates = [
        (
            "There is a {en} at {location_en} firing everywhere > F7C-S Hornet Ghost",
            "{location_zh}有个{zh}到处开火 >F7C-S Hornet Ghost",
        ),
        (
            "This feels bad; there is a {en} at {location_en} firing everywhere > F7C-S Hornet Ghost",
            "情况不太对，{location_zh}有个{zh}到处开火 >F7C-S Hornet Ghost",
        ),
        (
            "Global chat says there is a {en} at {location_en} shooting at players @...",
            "全局说{location_zh}有个{zh}在攻击玩家@...",
        ),
        (
            "Quick callout: {location_en} has a {en} spraying fire everywhere [global]",
            "报点 {location_zh}有个{zh}到处乱射[全局]",
        ),
        (
            "Warning: the {en} at {location_en} is shooting players > Drake Cutter",
            "注意 {location_zh}那艘{zh}在攻击玩家>Drake Cutter",
        ),
        (
            "Do not read the trailing tag as the ship; the ship at {location_en} is a {en} > F7C-S Hornet Ghost",
            "别把后面的标签当船名，{location_zh}那艘是{zh} >F7C-S Hornet Ghost",
        ),
    ]
    alias_chat_comm_templates = [
        (
            "We are taking the {en} from {location_en} for a bounty contract, then switching to cargo if the server stays stable.",
            "我们从{location_zh}开{zh}打赏金合同，如果服务器稳定就转去跑货。",
        ),
        (
            "The {en} near {location_en} is waiting for a medical rescue beacon; bring escort and do not shoot first.",
            "{location_zh}附近那艘{zh}在等医疗救援信标，带护航过来，先别开火。",
        ),
        (
            "Party chat says the {en} at {location_en} needs refuel, repair, and a turret seat before the next contract.",
            "队伍说{location_zh}那艘{zh}下一单合同前需要补油、维修和炮塔位。",
        ),
        (
            "Quick callout: the {en} at {location_en} is our cargo ship, not the bounty target > F7C-S Hornet Ghost",
            "报点 {location_zh}那艘{zh}是我们的货船，不是赏金目标 >F7C-S Hornet Ghost",
        ),
        (
            "Voice chat: keep the {en} outside {location_en} while we clear the bunker and loot the boxes.",
            "yy里说 我们清地堡和摸箱子的时候，让{zh}停在{location_zh}外面。",
        ),
        (
            "Need help at {location_en}; the {en} is in soft death, cargo is still on board, and pirates are boarding.",
            "{location_zh}需要支援，{zh}软死亡了，货还在船上，海盗正在登船。",
        ),
        (
            "Do not confuse the tag after the message; the ship asking for escort at {location_en} is the {en} @...",
            "别把消息后面的标签认成船名，在{location_zh}喊护航的是{zh}@...",
        ),
        (
            "New player note: if the {en} claim timer is long at {location_en}, ask for a pickup instead of buying a random ship.",
            "萌新注意 如果{zh}在{location_zh}的申领时间很长，就叫人接你，不要乱买船。",
        ),
        (
            "SC global says the {en} at {location_en} is mining nearby; keep the escort on it until the refinery run.",
            "全局说{location_zh}那艘{zh}在附近采矿，去精炼前继续护航。",
        ),
        (
            "If the {en} at {location_en} is doing salvage, mark the wreck and keep the cargo grid clear.",
            "如果{location_zh}那艘{zh}在打捞，标记残骸并把货物网格空出来。",
        ),
        (
            "Take the service beacon at {location_en} only if the {en} has escort and the payment looks right.",
            "{location_zh}的服务信标只有{zh}有护航而且报酬合适时才接。",
        ),
        (
            "The {en} crew is stuck by the elevator at {location_en}; keep the hangar clear until they get out.",
            "{zh}的船员卡在{location_zh}电梯旁边，出来前保持机库畅通。",
        ),
        (
            "Bring a tractor beam to {location_en}; the {en} has loose boxes on the cargo grid.",
            "带牵引光束来{location_zh}，{zh}的货物网格上有散货。",
        ),
    ]
    vehicle_comm_templates = [
        (
            "Party chat says the {en} at {location_en} is ready for the next bounty contract.",
            "队伍说{location_zh}那艘{zh}准备好接下一单赏金合同了。",
        ),
        (
            "We are taking the {en} from {location_en} for cargo hauling and need escort until the station.",
            "我们从{location_zh}开{zh}跑货，需要护航到空间站。",
        ),
        (
            "The {en} near {location_en} is waiting on a medical rescue beacon; do not shoot first.",
            "{location_zh}附近那艘{zh}在等医疗救援信标，先别开火。",
        ),
        (
            "Voice chat says the {en} at {location_en} needs refuel, repair, and quantum fuel before the jump.",
            "yy里说{location_zh}那艘{zh}跳跃前需要补油、维修和量子燃料。",
        ),
        (
            "SC global says the {en} near {location_en} is mining; keep escort until the refinery run.",
            "全局说{location_zh}附近那艘{zh}在采矿，去精炼前继续护航。",
        ),
        (
            "If the {en} at {location_en} is doing salvage, mark the wreck and keep the cargo grid clear.",
            "如果{location_zh}那艘{zh}在打捞，标记残骸并把货物网格空出来。",
        ),
        (
            "Take the service beacon at {location_en} only if the {en} has escort and the payment looks right.",
            "{location_zh}的服务信标只有{zh}有护航而且报酬合适时才接。",
        ),
        (
            "The {en} crew is stuck by the elevator at {location_en}; keep the hangar clear until they get out.",
            "{zh}的船员卡在{location_zh}电梯旁边，出来前保持机库畅通。",
        ),
        (
            "Bring a tractor beam to {location_en}; the {en} has loose boxes on the cargo grid.",
            "带牵引光束来{location_zh}，{zh}的货物网格上有散货。",
        ),
        (
            "Quick callout: the {en} at {location_en} is our ship, not the target after the tag > F7C-S Hornet Ghost",
            "报点 {location_zh}那艘{zh}是我们的船，不是后面标签里的目标 >F7C-S Hornet Ghost",
        ),
    ]
    vehicle_social_templates = [
        (
            "LFG bounty run in the {en} from {location_en}; anyone want to join?",
            "LFG 从{location_zh}开{zh}打赏金，有没有一起的？",
        ),
        (
            "I am running bounty missions in the {en} near {location_en}; need one gunner and one escort.",
            "我在{location_zh}附近开{zh}打赏金，缺一个炮手和一个护航。",
        ),
        (
            "Anyone up for a quick contract share? I can bring the {en} from {location_en}.",
            "有无一起共享个快速合同？我可以从{location_zh}开{zh}过去。",
        ),
        (
            "WTB cargo escort from {location_en}; the {en} is loaded and payment is on success.",
            "WTB 从{location_zh}出发的跑货护航，{zh}已经装货，成功后给报酬。",
        ),
        (
            "WTS salvage boxes at {location_en}; meet the {en} and bring a tractor beam.",
            "WTS {location_zh}的打捞箱子，到{zh}旁边集合，带牵引光束。",
        ),
        (
            "Need med rescue at {location_en}; the {en} is in soft death and pirates are boarding.",
            "{location_zh}需要医疗救援，{zh}软死亡了，海盗正在登船。",
        ),
        (
            "Party invite open for the {en} crew at {location_en}; plz join voice before launch.",
            "{location_zh}的{zh}船员队伍开放邀请，出发前plz进语音。",
        ),
        (
            "ASAP pickup near {location_en}; my {en} exploded and I still have the contract.",
            "{location_zh}附近ASAP来接，我的{zh}炸了，但合同还在。",
        ),
        (
            "o7, the {en} at {location_en} needs a turret player for high-risk bounty work.",
            "o7，{location_zh}那艘{zh}高风险赏金缺一个炮塔玩家。",
        ),
        (
            "Can someone cover the {en} from {location_en} to the station? I will split the beacon payment.",
            "有人能护送{zh}从{location_zh}去空间站吗？信标报酬我分。",
        ),
        (
            "New player in a {en} at {location_en}; looking for a group to learn cargo and bounty routes.",
            "萌新在{location_zh}开{zh}，找队伍学跑货和赏金路线。",
        ),
        (
            "[Party] LFG: {en} from {location_en}, bounty first then cargo if the server holds.",
            "[Party] LFG: 从{location_zh}开{zh}，先打赏金，服务器稳就跑货。",
        ),
    ]
    location_comm_ships = [
        ("Cutter", "小刀"),
        ("Razor", "剃刀"),
        ("Corsair", "海盗船"),
        ("Polaris", "北极星"),
        ("Perseus", "英仙座"),
        ("Scorpius", "天蝎座"),
        ("Prospector", "勘探者"),
        ("Vulture", "秃鹫"),
    ]
    location_comm_templates = [
        (
            "Meet at {en}; I am bringing the {ship_en} for a bounty contract.",
            "{zh}集合，我开{ship_zh}去打赏金合同。",
        ),
        (
            "Do not land at {en} yet; a {ship_en} is camping the hangar.",
            "先别降落{zh}，有艘{ship_zh}在蹲机库。",
        ),
        (
            "The cargo run from {en} needs escort, and the {ship_en} is already loaded.",
            "从{zh}出发的跑货需要护航，{ship_zh}已经装好货了。",
        ),
        (
            "Medical rescue is needed near {en}; the {ship_en} can pick up the beacon.",
            "{zh}附近需要医疗救援，{ship_zh}可以接信标。",
        ),
        (
            "SC global says {en} has heavy desync, so keep the {ship_en} away from the pad.",
            "全局说{zh}同步很差，让{ship_zh}先离停机坪远点。",
        ),
        (
            "If the {ship_en} reaches {en}, refuel, repair, and restock before the next jump.",
            "如果{ship_zh}到{zh}了，下一跳前先补油、维修和补弹药。",
        ),
        (
            "Quick callout: the target marker is at {en}, not on the ship tag after the message > F7C-S Hornet Ghost",
            "报点 目标标记在{zh}，不是消息后面那个船名标签 >F7C-S Hornet Ghost",
        ),
        (
            "The {ship_en} is mining near {en}; keep escort until the refinery job is ready.",
            "{ship_zh}在{zh}附近采矿，精炼任务准备好前继续护航。",
        ),
        (
            "The {ship_en} is doing salvage near {en}; mark the wreck and keep the cargo grid clear.",
            "{ship_zh}在{zh}附近打捞，标记残骸并把货物网格空出来。",
        ),
        (
            "A service beacon at {en} looks risky; bring the {ship_en} only after the party accepts.",
            "{zh}的服务信标看起来有风险，队伍接了以后再开{ship_zh}过去。",
        ),
        (
            "The elevator at {en} is bugged; keep the {ship_en} outside until the crew gets out.",
            "{zh}的电梯出问题了，让{ship_zh}在外面等到船员出来。",
        ),
        (
            "Bring a tractor beam to {en}; the {ship_en} has loose cargo boxes on the grid.",
            "带牵引光束来{zh}，{ship_zh}货物网格上有散货箱。",
        ),
    ]
    location_social_templates = [
        (
            "LFG at {en}; bringing a {ship_en} for bounty missions, anyone want to join?",
            "LFG {zh}集合，我开{ship_zh}打赏金，有没有一起的？",
        ),
        (
            "WTB escort from {en}; my {ship_en} has cargo and I can pay after delivery.",
            "WTB 从{zh}出发的护航，我的{ship_zh}有货，到货后给报酬。",
        ),
        (
            "WTS RMC near {en}; meet my {ship_en} and bring a tractor beam.",
            "WTS {zh}附近的RMC，到我的{ship_zh}旁边集合，带牵引光束。",
        ),
        (
            "Need med rescue near {en}; the {ship_en} is down and the beacon payment is shared.",
            "{zh}附近需要医疗救援，{ship_zh}倒了，信标报酬共享。",
        ),
        (
            "Party invite for {en}; plz join voice before we launch the {ship_en}.",
            "{zh}队伍邀请开放，开{ship_zh}出发前plz进语音。",
        ),
        (
            "ASAP pickup at {en}; contract is shared and the {ship_en} is waiting outside.",
            "{zh}ASAP来接，合同已共享，{ship_zh}在外面等。",
        ),
        (
            "o7, anyone at {en} want to run cargo, bunkers, or bounty contracts with a {ship_en}?",
            "o7，{zh}有人想一起开{ship_zh}跑货、清地堡或者打赏金吗？",
        ),
        (
            "[Global] LFG near {en}: need gunner, escort, and someone who can take a rescue beacon.",
            "[Global] {zh}附近LFG：缺炮手、护航，以及能接救援信标的人。",
        ),
    ]
    location_route_templates = [
        (
            "Route from {origin_en} to {destination_en}; bring the {ship_en} and keep escort on the cargo.",
            "从{origin_zh}去{destination_zh}，开{ship_zh}并继续护航货物。",
        ),
        (
            "Do not quantum from {origin_en} to {destination_en} until the {ship_en} refuels and repairs.",
            "{ship_zh}补油维修前，先别从{origin_zh}量子跳到{destination_zh}。",
        ),
        (
            "If {destination_en} has heavy desync, hold the {ship_en} at {origin_en} and ask in global chat.",
            "如果{destination_zh}同步很差，让{ship_zh}停在{origin_zh}并去全局问一下。",
        ),
        (
            "Medical rescue between {origin_en} and {destination_en}; the {ship_en} can take the service beacon.",
            "{origin_zh}到{destination_zh}之间有医疗救援，{ship_zh}可以接服务信标。",
        ),
        (
            "Move the salvage boxes from {origin_en} to {destination_en}; keep the {ship_en} cargo grid clear.",
            "把打捞箱子从{origin_zh}运到{destination_zh}，保持{ship_zh}货物网格清空。",
        ),
        (
            "Quick callout: target marker is on the route from {origin_en} to {destination_en}, not after the ship tag > F7C-S Hornet Ghost",
            "报点 目标标记在{origin_zh}到{destination_zh}的路线上，不是后面船名标签 >F7C-S Hornet Ghost",
        ),
    ]
    vehicle_contrast_templates = [
        (
            "I said {left_en}, not {right_en}; the ship at {location_en} is the one asking for escort.",
            "我说的是{left_zh}，不是{right_zh}；在{location_zh}喊护航的是它。",
        ),
        (
            "The bounty target is in the {left_en}, not the {right_en}; check the marker before firing.",
            "赏金目标在{left_zh}上，不在{right_zh}上，开火前看标记。",
        ),
        (
            "Use the {left_en} for this cargo run and keep the {right_en} parked at {location_en}.",
            "这趟跑货用{left_zh}，让{right_zh}停在{location_zh}。",
        ),
        (
            "Voice chat: the {left_en} needs refuel and repair, while the {right_en} needs a turret seat.",
            "yy里说 {left_zh}需要补油维修，{right_zh}缺炮塔位。",
        ),
        (
            "If the {left_en} goes into soft death near {location_en}, do not blame the {right_en} tag after the message.",
            "如果{left_zh}在{location_zh}附近软死亡了，别怪消息后面的{right_zh}标签。",
        ),
        (
            "Quick callout: {left_en} is our ship, {right_en} is just noise after the message > F7C-S Hornet Ghost",
            "报点 {left_zh}是我们的船，{right_zh}只是消息后面的噪声 >F7C-S Hornet Ghost",
        ),
    ]
    vehicle_mixed_format_templates = [
        (
            "[party] @teammate The {en} at pad 03 in {location_en} is in soft death >>> hold fire.",
            "[队伍] @队友 {location_zh} pad 03 有{zh}软死了 >>> 别开火",
        ),
        (
            "SC global: {en} from {location_en} to {destination_en} is hauling cargo; need escort?",
            "全局: {zh}@{location_zh}->{destination_zh} 跑货，需要护航？",
        ),
        (
            "Near OM-1, the {en} is locking missiles @... check marker.",
            "OM-1附近 {zh}在锁导弹 @... 看标记",
        ),
        (
            "Voice: {en} needs refuel and repair at {location_en}; do not read > F7C-S Hornet Ghost as the ship.",
            "yy {zh}在{location_zh}要补油维修，别把 >F7C-S Hornet Ghost 当船名",
        ),
    ]
    location_mixed_format_templates = [
        (
            "[global] @teammate A {ship_en} is blocking pad 03 at {en} >>> switch route.",
            "[全局] @队友 {zh} pad 03 有{ship_zh}堵门 >>> 换route",
        ),
        (
            "{en} to OM-1 has a medical beacon; can the {ship_en} take it?",
            "{zh}->OM-1 有救援信标，{ship_zh}接吗？",
        ),
        (
            "Voice: elevator bug at {en}; do not land the {ship_en}.",
            "yy {zh} elevator bug，{ship_zh}别landing",
        ),
        (
            "Quick callout: marker 2 is at {en}, not the tag after @...",
            "报点 marker 2在{zh}，不是@...后面的标签",
        ),
    ]
    vehicle_chat_log_templates = [
        (
            "[Global][RedNine]: anyone near {location_en}?\n[Party][Me]: the {en} is in soft death at pad 03, hold fire\n[Voice][Kai]: mark it before boarding",
            "[Global][RedNine]: {location_zh}附近有人吗？\n[Party][Me]: {zh}在pad 03软死了，先别开火\n[Voice][Kai]: 登船前先标记",
        ),
        (
            "21:04 [Party] Mira: taking the {en} from {location_en} to {destination_en}\n21:05 [Party] Sol: cargo grid clear?\n21:05 [Party] Mira: clear, need escort",
            "21:04 [Party] Mira: 从{location_zh}开{zh}去{destination_zh}\n21:05 [Party] Sol: 货物网格清了吗？\n21:05 [Party] Mira: 清了，需要护航",
        ),
        (
            "[Local] UEE: service beacon accepted\n[Global] Fox: {en} near OM-1 is locking missiles @...\n[Party] Me: check marker, do not read the tag as the ship",
            "[Local] UEE: 服务信标已接\n[Global] Fox: OM-1附近{zh}在锁导弹 @...\n[Party] Me: 看标记，别把标签当船名",
        ),
    ]
    location_chat_log_templates = [
        (
            "[Global][Ari]: status at {en}?\n[Party][Me]: elevator bug, keep the {ship_en} outside\n[Party][Ren]: copy, no landing yet",
            "[Global][Ari]: {zh}什么情况？\n[Party][Me]: elevator bug，让{ship_zh}在外面等\n[Party][Ren]: 收到，先不landing",
        ),
        (
            "20:11 [Team] Lyn: route {en} -> OM-1 has a med beacon\n20:12 [Team] Vox: bring {ship_en}?\n20:12 [Team] Lyn: yes, but wait for escort",
            "20:11 [Team] Lyn: {zh}->OM-1 有救援信标\n20:12 [Team] Vox: 开{ship_zh}？\n20:12 [Team] Lyn: 对，但等护航",
        ),
        (
            "[Voice] Kai: marker 2 is at {en}\n[Global] Nova: I only see > F7C-S Hornet Ghost\n[Voice] Kai: ignore that tag, follow marker 2",
            "[Voice] Kai: marker 2在{zh}\n[Global] Nova: 我只看到 >F7C-S Hornet Ghost\n[Voice] Kai: 忽略那个标签，跟marker 2",
        ),
    ]
    alias_chat_log_templates = [
        (
            "[Global][RedNine]: anyone at {location_en}?\n[Party][Me]: {en} is in soft death, cargo still on board\n[Voice][Kai]: board after marker",
            "[Global][RedNine]: {location_zh}有人吗？\n[Party][Me]: {zh}软死了，货还在船上\n[Voice][Kai]: 标记后再登船",
        ),
        (
            "21:04 [Party] Mira: {en} from {location_en} is asking for escort\n21:05 [Party] Sol: target or friendly?\n21:05 [Party] Mira: friendly, do not shoot",
            "21:04 [Party] Mira: {location_zh}那艘{zh}在喊escort\n21:05 [Party] Sol: 目标还是友军？\n21:05 [Party] Mira: 友军，别开火",
        ),
    ]
    gameplay_comm_templates = [
        (
            "Party chat: this is {en}, not a ship name; keep the {ship_en} at {location_en} until we confirm it.",
            "队伍说 这是{zh}，不是船名；确认前让{ship_zh}停在{location_zh}。",
        ),
        (
            "[Global] Fox: {en} near {location_en} @...\n[Party] Me: translate that as {en}, not as a random ship\n[Voice] Kai: check marker before firing",
            "[Global] Fox: {location_zh}附近{zh} @...\n[Party] Me: 这个要翻成{en}，不是随机船名\n[Voice] Kai: 开火前看标记",
        ),
        (
            "Quick callout: {en} on the {ship_en} at {location_en}; do not confuse it with > F7C-S Hornet Ghost.",
            "报点 {location_zh}那艘{ship_zh}是{zh}，别和 >F7C-S Hornet Ghost 混了。",
        ),
        (
            "Voice: if players say this term, use {en}; keep the party on voice and wait for escort.",
            "yy 玩家说{zh}就用{en}，队伍保持语音并等护航。",
        ),
    ]
    gameplay_social_templates = [
        (
            "LFG: this is about {en}; share the contract and wait for everyone to accept.",
            "LFG: 这里说的是{zh}；共享合同并等所有人接了再走。",
        ),
        (
            "WTB help with {en} near {location_en}; payment after the beacon completes.",
            "WTB {location_zh}附近{zh}帮忙，信标完成后给报酬。",
        ),
        (
            "Need someone on voice for {en}; plz do not translate it as a ship.",
            "{zh}需要有人上语音，plz别把它翻成船名。",
        ),
        (
            "ASAP party invite for {en}; the {ship_en} is waiting at {location_en}.",
            "{zh}ASAP进队，{ship_zh}在{location_zh}等。",
        ),
        (
            "o7, if global chat says {en}, keep that gameplay term and do not replace it with a vehicle.",
            "o7，如果全局说{zh}，保留这个玩法术语，别替换成载具。",
        ),
        (
            "[Party] LFG for {en}: need escort, rescue, and someone to watch the marker.",
            "[Party] {zh} LFG：缺护航、救援，以及盯标记的人。",
        ),
    ]
    alias_chat_slang_prefixes = [
        ("SC global: ", "全局 "),
        ("Party: ", "队伍 "),
        ("Voice: ", "yy里 "),
        ("Need help: ", "来人 "),
        ("This feels bad; ", "情况不太对 "),
        ("Newbie warning: ", "萌新注意 "),
        ("Quick callout: ", "报点 "),
    ]
    alias_chat_slang_suffixes = [
        (" ASAP.", " 速来"),
        (" Anyone up?", " 有人吗"),
        (" Check marker.", " 看标记"),
        (" Stay on voice.", " 进语音"),
        (" @...", " @..."),
        (" [global]", " [全局]"),
        (" [voice]", " [语音]"),
    ]
    alias_chat_slang_replacements = [
        ("打赏金", "刷赏金"),
        ("软死亡", "软死"),
        ("量子燃料", "量子油"),
        ("医疗信标", "救援信标"),
        ("医疗救援", "医疗救援"),
        ("地堡任务", "地堡"),
        ("申领时间", "申领倒计时"),
        ("同步很差", "同步炸了"),
        ("炮塔位", "炮塔位"),
        ("犯罪等级", "罪等"),
        ("赏金目标", "赏金目标"),
        ("赏金合同", "赏金单"),
        ("医疗救援信标", "救援信标"),
        ("补油维修", "补油修船"),
    ]

    def compact_alias_chat(text: str) -> str:
        return (
            text.replace("。", " ")
            .replace("？", " ")
            .replace("！", " ")
            .replace("，", " ")
            .replace("；", " ")
            .replace("：", " ")
            .replace("  ", " ")
            .strip()
        )

    def strip_final_punctuation(text: str) -> str:
        return text.rstrip().rstrip(".?!")

    def slangify_chat(text: str) -> str:
        slang_text = compact_alias_chat(text)
        for source_phrase, replacement in sorted(
            alias_chat_slang_replacements,
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            slang_text = slang_text.replace(source_phrase, replacement)
        return slang_text

    term_entry_list = list(term_entries)
    alias_entry_list = list(alias_entries)
    formal_vehicle_entries = [entry for entry in term_entry_list if entry.category == "vehicle"]
    formal_location_entries = [entry for entry in term_entry_list if entry.category == "location"]
    gameplay_entries = [entry for entry in term_entry_list if entry.category == "gameplay"]
    entries = term_entry_list + alias_entry_list
    seen_entries: set[tuple[str, str, str]] = set()
    for entry_index, entry in enumerate(entries, start=1):
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
            if entry.category == "vehicle" and not entry.key.startswith("ship_alias:"):
                for location_index, (location_en, location_zh) in enumerate(alias_chat_locations, start=1):
                    for template_index, (en_template, zh_template) in enumerate(vehicle_comm_templates, start=1):
                        en_text = en_template.format(
                            en=entry.en,
                            zh=entry.zh,
                            location_en=location_en,
                            location_zh=location_zh,
                        )
                        zh_text = zh_template.format(
                            en=entry.en,
                            zh=entry.zh,
                            location_en=location_en,
                            location_zh=location_zh,
                        )
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_vehicle_comm:{entry.key}:{repeat_index + 1}:"
                                    f"{location_index}:{template_index}:standard"
                                ),
                                en=en_text,
                                zh=zh_text,
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
                        slang_zh = compact_alias_chat(zh_text)
                        for source_phrase, replacement in sorted(
                            alias_chat_slang_replacements,
                            key=lambda item: len(item[0]),
                            reverse=True,
                        ):
                            slang_zh = slang_zh.replace(source_phrase, replacement)
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_vehicle_comm:{entry.key}:{repeat_index + 1}:"
                                    f"{location_index}:{template_index}:slang"
                                ),
                                en=en_text,
                                zh=slang_zh,
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
                current_vehicle_index = formal_vehicle_entries.index(entry) if entry in formal_vehicle_entries else entry_index
                for template_index, (en_template, zh_template) in enumerate(vehicle_social_templates, start=1):
                    location_en, location_zh = alias_chat_locations[
                        (current_vehicle_index + template_index + repeat_index) % len(alias_chat_locations)
                    ]
                    en_text = en_template.format(
                        en=entry.en,
                        zh=entry.zh,
                        location_en=location_en,
                        location_zh=location_zh,
                    )
                    zh_text = zh_template.format(
                        en=entry.en,
                        zh=entry.zh,
                        location_en=location_en,
                        location_zh=location_zh,
                    )
                    for style, source_text in (("standard", zh_text), ("slang", slangify_chat(zh_text))):
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_vehicle_social:{entry.key}:{repeat_index + 1}:"
                                    f"{template_index}:{style}"
                                ),
                                en=en_text,
                                zh=source_text,
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
                if formal_vehicle_entries:
                    current_vehicle_index = formal_vehicle_entries.index(entry)
                    contrast_offsets = (1, 7)
                    for contrast_index, offset in enumerate(contrast_offsets, start=1):
                        right_index = (current_vehicle_index + offset) % len(formal_vehicle_entries)
                        right_entry = formal_vehicle_entries[right_index]
                        attempts = 0
                        while (
                            (
                                right_entry == entry
                                or right_entry.en.casefold() == entry.en.casefold()
                                or right_entry.zh == entry.zh
                            )
                            and attempts < len(formal_vehicle_entries)
                        ):
                            right_index = (right_index + 1) % len(formal_vehicle_entries)
                            right_entry = formal_vehicle_entries[right_index]
                            attempts += 1
                        if right_entry == entry or right_entry.en.casefold() == entry.en.casefold():
                            continue
                        location_en, location_zh = alias_chat_locations[
                            (current_vehicle_index + contrast_index + repeat_index) % len(alias_chat_locations)
                        ]
                        for template_index, (en_template, zh_template) in enumerate(vehicle_contrast_templates, start=1):
                            en_text = en_template.format(
                                left_en=entry.en,
                                left_zh=entry.zh,
                                right_en=right_entry.en,
                                right_zh=right_entry.zh,
                                location_en=location_en,
                                location_zh=location_zh,
                            )
                            zh_text = zh_template.format(
                                left_en=entry.en,
                                left_zh=entry.zh,
                                right_en=right_entry.en,
                                right_zh=right_entry.zh,
                                location_en=location_en,
                                location_zh=location_zh,
                            )
                            samples.append(
                                PairSample(
                                    key=(
                                        f"quant_focus_vehicle_contrast:{entry.key}:{repeat_index + 1}:"
                                        f"{contrast_index}:{template_index}:standard"
                                    ),
                                    en=en_text,
                                    zh=zh_text,
                                    category=entry.category,
                                    is_priority=True,
                                    source="quant_focus",
                                )
                            )
                            slang_zh = compact_alias_chat(zh_text)
                            for source_phrase, replacement in sorted(
                                alias_chat_slang_replacements,
                                key=lambda item: len(item[0]),
                                reverse=True,
                            ):
                                slang_zh = slang_zh.replace(source_phrase, replacement)
                            samples.append(
                                PairSample(
                                    key=(
                                        f"quant_focus_vehicle_contrast:{entry.key}:{repeat_index + 1}:"
                                        f"{contrast_index}:{template_index}:slang"
                                    ),
                                    en=en_text,
                                    zh=slang_zh,
                                    category=entry.category,
                                    is_priority=True,
                                    source="quant_focus",
                                )
                            )
                current_vehicle_index = formal_vehicle_entries.index(entry) if entry in formal_vehicle_entries else entry_index
                for template_index, (en_template, zh_template) in enumerate(vehicle_mixed_format_templates, start=1):
                    location_en, location_zh = alias_chat_locations[
                        (current_vehicle_index + template_index + repeat_index) % len(alias_chat_locations)
                    ]
                    destination_en, destination_zh = alias_chat_locations[
                        (current_vehicle_index + template_index + repeat_index + 3) % len(alias_chat_locations)
                    ]
                    if destination_en == location_en:
                        destination_en, destination_zh = alias_chat_locations[
                            (current_vehicle_index + template_index + repeat_index + 4) % len(alias_chat_locations)
                        ]
                    samples.append(
                        PairSample(
                            key=(
                                f"quant_focus_vehicle_mixed_format:{entry.key}:{repeat_index + 1}:"
                                f"{template_index}"
                            ),
                            en=en_template.format(
                                en=entry.en,
                                zh=entry.zh,
                                location_en=location_en,
                                location_zh=location_zh,
                                destination_en=destination_en,
                                destination_zh=destination_zh,
                            ),
                            zh=zh_template.format(
                                en=entry.en,
                                zh=entry.zh,
                                location_en=location_en,
                                location_zh=location_zh,
                                destination_en=destination_en,
                                destination_zh=destination_zh,
                            ),
                            category=entry.category,
                            is_priority=True,
                            source="quant_focus",
                        )
                    )
                for template_index, (en_template, zh_template) in enumerate(vehicle_chat_log_templates, start=1):
                    location_en, location_zh = alias_chat_locations[
                        (current_vehicle_index + template_index + repeat_index) % len(alias_chat_locations)
                    ]
                    destination_en, destination_zh = alias_chat_locations[
                        (current_vehicle_index + template_index + repeat_index + 2) % len(alias_chat_locations)
                    ]
                    if destination_en == location_en:
                        destination_en, destination_zh = alias_chat_locations[
                            (current_vehicle_index + template_index + repeat_index + 3) % len(alias_chat_locations)
                        ]
                    samples.append(
                        PairSample(
                            key=(
                                f"quant_focus_vehicle_chat_log:{entry.key}:{repeat_index + 1}:"
                                f"{template_index}"
                            ),
                            en=en_template.format(
                                en=entry.en,
                                zh=entry.zh,
                                location_en=location_en,
                                location_zh=location_zh,
                                destination_en=destination_en,
                                destination_zh=destination_zh,
                            ),
                            zh=zh_template.format(
                                en=entry.en,
                                zh=entry.zh,
                                location_en=location_en,
                                location_zh=location_zh,
                                destination_en=destination_en,
                                destination_zh=destination_zh,
                            ),
                            category=entry.category,
                            is_priority=True,
                            source="quant_focus",
                        )
                    )
            if entry.category == "location":
                current_location_index = formal_location_entries.index(entry) if entry in formal_location_entries else entry_index
                if formal_location_entries:
                    route_offsets = (1, 9, 31)
                    for route_index, offset in enumerate(route_offsets, start=1):
                        destination_index = (current_location_index + offset) % len(formal_location_entries)
                        destination_entry = formal_location_entries[destination_index]
                        attempts = 0
                        while (
                            (
                                destination_entry == entry
                                or destination_entry.en.casefold() == entry.en.casefold()
                                or destination_entry.zh == entry.zh
                            )
                            and attempts < len(formal_location_entries)
                        ):
                            destination_index = (destination_index + 1) % len(formal_location_entries)
                            destination_entry = formal_location_entries[destination_index]
                            attempts += 1
                        if destination_entry == entry or destination_entry.en.casefold() == entry.en.casefold():
                            continue
                        ship_en, ship_zh = location_comm_ships[
                            (current_location_index + route_index + repeat_index) % len(location_comm_ships)
                        ]
                        for template_index, (en_template, zh_template) in enumerate(location_route_templates, start=1):
                            en_text = en_template.format(
                                origin_en=entry.en,
                                origin_zh=entry.zh,
                                destination_en=destination_entry.en,
                                destination_zh=destination_entry.zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            )
                            zh_text = zh_template.format(
                                origin_en=entry.en,
                                origin_zh=entry.zh,
                                destination_en=destination_entry.en,
                                destination_zh=destination_entry.zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                            )
                            samples.append(
                                PairSample(
                                    key=(
                                        f"quant_focus_location_route:{entry.key}:{repeat_index + 1}:"
                                        f"{route_index}:{template_index}:standard"
                                    ),
                                    en=en_text,
                                    zh=zh_text,
                                    category=entry.category,
                                    is_priority=True,
                                    source="quant_focus",
                                )
                            )
                            slang_zh = compact_alias_chat(zh_text)
                            for source_phrase, replacement in sorted(
                                alias_chat_slang_replacements,
                                key=lambda item: len(item[0]),
                                reverse=True,
                            ):
                                slang_zh = slang_zh.replace(source_phrase, replacement)
                            samples.append(
                                PairSample(
                                    key=(
                                        f"quant_focus_location_route:{entry.key}:{repeat_index + 1}:"
                                        f"{route_index}:{template_index}:slang"
                                    ),
                                    en=en_text,
                                    zh=slang_zh,
                                    category=entry.category,
                                    is_priority=True,
                                    source="quant_focus",
                                )
                            )
                for template_index, (en_template, zh_template) in enumerate(location_mixed_format_templates, start=1):
                    ship_en, ship_zh = location_comm_ships[
                        (current_location_index + template_index + repeat_index) % len(location_comm_ships)
                    ]
                    samples.append(
                        PairSample(
                            key=(
                                f"quant_focus_location_mixed_format:{entry.key}:{repeat_index + 1}:"
                                f"{template_index}"
                            ),
                            en=en_template.format(en=entry.en, zh=entry.zh, ship_en=ship_en, ship_zh=ship_zh),
                            zh=zh_template.format(en=entry.en, zh=entry.zh, ship_en=ship_en, ship_zh=ship_zh),
                            category=entry.category,
                            is_priority=True,
                            source="quant_focus",
                        )
                    )
                for template_index, (en_template, zh_template) in enumerate(location_chat_log_templates, start=1):
                    ship_en, ship_zh = location_comm_ships[
                        (current_location_index + template_index + repeat_index + 2) % len(location_comm_ships)
                    ]
                    samples.append(
                        PairSample(
                            key=(
                                f"quant_focus_location_chat_log:{entry.key}:{repeat_index + 1}:"
                                f"{template_index}"
                            ),
                            en=en_template.format(en=entry.en, zh=entry.zh, ship_en=ship_en, ship_zh=ship_zh),
                            zh=zh_template.format(en=entry.en, zh=entry.zh, ship_en=ship_en, ship_zh=ship_zh),
                            category=entry.category,
                            is_priority=True,
                            source="quant_focus",
                        )
                    )
                for template_index, (en_template, zh_template) in enumerate(location_social_templates, start=1):
                    ship_en, ship_zh = location_comm_ships[
                        (current_location_index + template_index + repeat_index + 4) % len(location_comm_ships)
                    ]
                    en_text = en_template.format(en=entry.en, zh=entry.zh, ship_en=ship_en, ship_zh=ship_zh)
                    zh_text = zh_template.format(en=entry.en, zh=entry.zh, ship_en=ship_en, ship_zh=ship_zh)
                    for style, source_text in (("standard", zh_text), ("slang", slangify_chat(zh_text))):
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_location_social:{entry.key}:{repeat_index + 1}:"
                                    f"{template_index}:{style}"
                                ),
                                en=en_text,
                                zh=source_text,
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
                for ship_index, (ship_en, ship_zh) in enumerate(location_comm_ships, start=1):
                    for template_index, (en_template, zh_template) in enumerate(location_comm_templates, start=1):
                        en_text = en_template.format(en=entry.en, zh=entry.zh, ship_en=ship_en, ship_zh=ship_zh)
                        zh_text = zh_template.format(en=entry.en, zh=entry.zh, ship_en=ship_en, ship_zh=ship_zh)
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_location_comm:{entry.key}:{repeat_index + 1}:"
                                    f"{ship_index}:{template_index}:standard"
                                ),
                                en=en_text,
                                zh=zh_text,
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
                        slang_zh = compact_alias_chat(zh_text)
                        for source_phrase, replacement in sorted(
                            alias_chat_slang_replacements,
                            key=lambda item: len(item[0]),
                            reverse=True,
                        ):
                            slang_zh = slang_zh.replace(source_phrase, replacement)
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_location_comm:{entry.key}:{repeat_index + 1}:"
                                    f"{ship_index}:{template_index}:slang"
                                ),
                                en=en_text,
                                zh=slang_zh,
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                        )
                    )
            if entry.category == "gameplay":
                gameplay_index = gameplay_entries.index(entry) if entry in gameplay_entries else entry_index
                for template_index, (en_template, zh_template) in enumerate(gameplay_comm_templates, start=1):
                    ship_en, ship_zh = location_comm_ships[
                        (gameplay_index + template_index + repeat_index) % len(location_comm_ships)
                    ]
                    location_en, location_zh = alias_chat_locations[
                        (gameplay_index + template_index + repeat_index) % len(alias_chat_locations)
                    ]
                    samples.append(
                        PairSample(
                            key=f"quant_focus_gameplay_comm:{entry.key}:{repeat_index + 1}:{template_index}",
                            en=en_template.format(
                                en=entry.en,
                                zh=entry.zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                                location_en=location_en,
                                location_zh=location_zh,
                            ),
                            zh=zh_template.format(
                                en=entry.en,
                                zh=entry.zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                                location_en=location_en,
                                location_zh=location_zh,
                            ),
                            category=entry.category,
                            is_priority=True,
                            source="quant_focus",
                        )
                    )
                for template_index, (en_template, zh_template) in enumerate(gameplay_social_templates, start=1):
                    ship_en, ship_zh = location_comm_ships[
                        (gameplay_index + template_index + repeat_index + 3) % len(location_comm_ships)
                    ]
                    location_en, location_zh = alias_chat_locations[
                        (gameplay_index + template_index + repeat_index + 3) % len(alias_chat_locations)
                    ]
                    en_text = en_template.format(
                        en=entry.en,
                        zh=entry.zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                        location_en=location_en,
                        location_zh=location_zh,
                    )
                    zh_text = zh_template.format(
                        en=entry.en,
                        zh=entry.zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                        location_en=location_en,
                        location_zh=location_zh,
                    )
                    for style, source_text in (("standard", zh_text), ("slang", slangify_chat(zh_text))):
                        samples.append(
                            PairSample(
                                key=f"quant_focus_gameplay_social:{entry.key}:{repeat_index + 1}:{template_index}:{style}",
                                en=en_text,
                                zh=source_text,
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
        if entry.key.startswith("ship_alias:") and entry.category == "vehicle":
            location_count = len(alias_chat_locations)
            selected_locations = [
                alias_chat_locations[(entry_index - 1) % location_count],
                alias_chat_locations[(entry_index + 2) % location_count],
            ]
            for repeat_index in range(max(1, alias_repeat)):
                for location_index, (location_en, location_zh) in enumerate(selected_locations, start=1):
                    for template_index, (en_template, zh_template) in enumerate(alias_chat_templates, start=1):
                        en_text = en_template.format(en=entry.en, zh=entry.zh, location_en=location_en, location_zh=location_zh)
                        zh_text = zh_template.format(en=entry.en, zh=entry.zh, location_en=location_en, location_zh=location_zh)
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_alias_chat:{entry.key}:{repeat_index + 1}:"
                                    f"{location_index}:{template_index}"
                                ),
                                en=en_text,
                                zh=zh_text,
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
                        slang_prefix_en, slang_prefix_zh = alias_chat_slang_prefixes[
                            (entry_index + template_index + repeat_index) % len(alias_chat_slang_prefixes)
                        ]
                        slang_suffix_en, slang_suffix_zh = alias_chat_slang_suffixes[
                            (entry_index + location_index + template_index) % len(alias_chat_slang_suffixes)
                        ]
                        slang_zh = compact_alias_chat(zh_text)
                        for source_phrase, replacement in sorted(
                            alias_chat_slang_replacements,
                            key=lambda item: len(item[0]),
                            reverse=True,
                        ):
                            slang_zh = slang_zh.replace(source_phrase, replacement)
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_alias_slang:{entry.key}:{repeat_index + 1}:"
                                    f"{location_index}:{template_index}"
                                ),
                                en=f"{slang_prefix_en}{strip_final_punctuation(en_text)}.{slang_suffix_en}",
                                zh=f"{slang_prefix_zh}{slang_zh}{slang_suffix_zh}",
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
                for location_index, (location_en, location_zh) in enumerate(selected_locations, start=1):
                    for template_index, (en_template, zh_template) in enumerate(vehicle_social_templates, start=1):
                        en_text = en_template.format(
                            en=entry.en,
                            zh=entry.zh,
                            location_en=location_en,
                            location_zh=location_zh,
                        )
                        zh_text = zh_template.format(
                            en=entry.en,
                            zh=entry.zh,
                            location_en=location_en,
                            location_zh=location_zh,
                        )
                        for style, source_text in (("standard", zh_text), ("slang", slangify_chat(zh_text))):
                            samples.append(
                                PairSample(
                                    key=(
                                        f"quant_focus_alias_social:{entry.key}:{repeat_index + 1}:"
                                        f"{location_index}:{template_index}:{style}"
                                    ),
                                    en=en_text,
                                    zh=source_text,
                                    category=entry.category,
                                    is_priority=True,
                                    source="quant_focus",
                                )
                            )
                for location_index, (location_en, location_zh) in enumerate(alias_chat_locations, start=1):
                    for template_index, (en_template, zh_template) in enumerate(alias_chat_noise_templates, start=1):
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_alias_noise:{entry.key}:{repeat_index + 1}:"
                                    f"{location_index}:{template_index}"
                                ),
                                en=en_template.format(
                                    en=entry.en,
                                    zh=entry.zh,
                                    location_en=location_en,
                                    location_zh=location_zh,
                                ),
                                zh=zh_template.format(
                                    en=entry.en,
                                    zh=entry.zh,
                                    location_en=location_en,
                                    location_zh=location_zh,
                                ),
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
                    for template_index, (en_template, zh_template) in enumerate(alias_chat_comm_templates, start=1):
                        en_text = en_template.format(
                            en=entry.en,
                            zh=entry.zh,
                            location_en=location_en,
                            location_zh=location_zh,
                        )
                        zh_text = zh_template.format(
                            en=entry.en,
                            zh=entry.zh,
                            location_en=location_en,
                            location_zh=location_zh,
                        )
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_alias_comm:{entry.key}:{repeat_index + 1}:"
                                    f"{location_index}:{template_index}:standard"
                                ),
                                en=en_text,
                                zh=zh_text,
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
                        slang_zh = compact_alias_chat(zh_text)
                        for source_phrase, replacement in sorted(
                            alias_chat_slang_replacements,
                            key=lambda item: len(item[0]),
                            reverse=True,
                        ):
                            slang_zh = slang_zh.replace(source_phrase, replacement)
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_alias_comm:{entry.key}:{repeat_index + 1}:"
                                    f"{location_index}:{template_index}:slang"
                                ),
                                en=en_text,
                                zh=slang_zh,
                                category=entry.category,
                                is_priority=True,
                                source="quant_focus",
                            )
                        )
                    for template_index, (en_template, zh_template) in enumerate(alias_chat_log_templates, start=1):
                        samples.append(
                            PairSample(
                                key=(
                                    f"quant_focus_alias_chat_log:{entry.key}:{repeat_index + 1}:"
                                    f"{location_index}:{template_index}"
                                ),
                                en=en_template.format(
                                    en=entry.en,
                                    zh=entry.zh,
                                    location_en=location_en,
                                    location_zh=location_zh,
                                ),
                                zh=zh_template.format(
                                    en=entry.en,
                                    zh=entry.zh,
                                    location_en=location_en,
                                    location_zh=location_zh,
                                ),
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
    gameplay_direct_templates = [
        ("{term_en}", "{term_zh}"),
        ("Status: {term_en}.", "状态：{term_zh}。"),
        ("Chat term: {term_en}.", "聊天术语：{term_zh}。"),
        ("The player said {term_en}.", "玩家说的是{term_zh}。"),
        ("This message is about {term_en}.", "这句话说的是{term_zh}。"),
        ("In this context, use {term_en}.", "这个语境里用{term_zh}。"),
        ("Do not replace {term_en} with a random ship name.", "不要把{term_zh}替换成随机船名。"),
        ("Keep the gameplay phrase as {term_en}.", "这个玩法词保留为{term_zh}对应的术语。"),
    ]
    chat_prefix_wrappers = [
        ("Global chat: ", "全局频道："),
        ("Voice chat: ", "语音里说："),
        ("Someone in party chat said: ", "队伍频道有人说："),
        ("Help, ", "救命，"),
        ("Warning, ", "注意，"),
        ("This feels bad; ", "情况不太对，"),
        ("Can someone confirm this? ", "谁确认一下，"),
        ("I just logged in and saw this: ", "我刚上线看到，"),
        ("Do not panic, but ", "先别慌，"),
        ("For the new players: ", "给新手说一下，"),
    ]
    chat_suffix_wrappers = [
        (" Anyone want to join?", "有没有一起的？"),
        (" Can anyone help?", "有人能帮忙吗？"),
        (" Please confirm before firing.", "开火前确认一下。"),
        (" I am not sure if it is friendly.", "我不确定是不是友军。"),
        (" Tell the party on voice.", "在语音里告诉队伍。"),
        (" Mark it before we engage.", "开打前先标记一下。"),
        (" New players should stay away.", "新手先离远点。"),
        (" Do not confuse the ship name.", "别把船名认错。"),
    ]
    chat_noise_suffixes = [
        (" > F7C-S Hornet Ghost", ">F7C-S Hornet Ghost"),
        (" > Aegis Gladius", ">Aegis Gladius"),
        (" > Drake Cutter", ">Drake Cutter"),
        (" @...", "@..."),
        (" [global]", "[全局]"),
        (" [voice]", "[语音]"),
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
    structured_chat_openers = [
        ("Global chat: ", "全局频道："),
        ("Party chat: ", "队伍频道："),
        ("Voice chat: ", "语音里说："),
        ("Warning: ", "警告："),
        ("Requesting help: ", "求支援："),
        ("For new players: ", "给新手说一下："),
        ("Can someone confirm this? ", "谁确认一下："),
        ("This feels bad; ", "情况不太对，"),
    ]
    structured_chat_followups = [
        (" Anyone want to join?", "有没有一起的？"),
        (" Can someone help?", "有人能帮忙吗？"),
        (" Do not shoot until we identify it.", "确认身份前先别开火。"),
        (" Mark it for the party.", "给队伍标记一下。"),
        (" Stay away from the pad.", "先离停机坪远点。"),
        (" Tell the new players in global chat.", "在全局频道提醒一下新手。"),
        (" I am not sure if it is friendly.", "我不确定是不是友军。"),
        (" Do not confuse the ship name.", "别把船名认错。"),
    ]
    structured_chat_events = [
        (
            "the {ship_en} at {location_en} is firing everywhere",
            "{location_zh}有个{ship_zh}到处开火",
        ),
        (
            "the {ship_en} near {location_en} is camping the hangar",
            "{location_zh}附近有个{ship_zh}在蹲机库",
        ),
        (
            "I am taking the {ship_en} from {location_en} for bounty missions",
            "我从{location_zh}开{ship_zh}打赏金",
        ),
        (
            "I am cargo hauling in the {ship_en} from {location_en}",
            "我从{location_zh}开{ship_zh}跑货",
        ),
        (
            "the {ship_en} at {location_en} is in soft death",
            "{location_zh}那艘{ship_zh}软死亡了",
        ),
        (
            "the {ship_en} at {location_en} has a bounty target on board",
            "{location_zh}那艘{ship_zh}上有赏金目标",
        ),
        (
            "the {ship_en} near {location_en} is locking missiles",
            "{location_zh}附近那艘{ship_zh}在锁导弹",
        ),
        (
            "the {ship_en} at {location_en} needs a gunner and a turret seat",
            "{location_zh}那艘{ship_zh}缺炮手和炮塔位",
        ),
        (
            "the {ship_en} at {location_en} needs quantum fuel before we jump",
            "{location_zh}那艘{ship_zh}跳跃前需要量子燃料",
        ),
        (
            "the {ship_en} at {location_en} is not a Hornet Ghost",
            "{location_zh}那艘{ship_zh}不是大黄蜂幽灵",
        ),
        (
            "I dropped a medical beacon near {location_en}; the {ship_en} can land there",
            "我在{location_zh}附近发了医疗信标，{ship_zh}能在那里降落",
        ),
        (
            "medical rescue is needed near {location_en}; bring the {ship_en}",
            "{location_zh}附近需要医疗救援，把{ship_zh}开过来",
        ),
        (
            "we are doing a bunker mission near {location_en}; leave the {ship_en} outside",
            "我们在{location_zh}附近做地堡任务，{ship_zh}停外面",
        ),
        (
            "the {ship_en} claim timer at {location_en} is almost done",
            "{ship_zh}在{location_zh}的申领时间快好了",
        ),
        (
            "{location_en} has heavy desync; the {ship_en} may rubber-band",
            "{location_zh}同步很差，{ship_zh}可能会来回瞬移",
        ),
    ]
    structured_server_events = [
        (
            "On the {server_en}, {event_en}",
            "{server_zh}{event_zh}",
        ),
        (
            "Reported on the {server_en}: {event_en}",
            "{server_zh}这边{event_zh}",
        ),
    ]
    structured_noise_pairs = [
        ("", ""),
        (" > F7C-S Hornet Ghost", ">F7C-S Hornet Ghost"),
        (" @...", "@..."),
        (" [global]", "[全局]"),
    ]
    structured_compound_actions = [
        ("I will bring the {ship_en} and engage first", "我开{ship_zh}先上"),
        ("wait for the party before opening fire", "等队伍到了再开火"),
        ("mark the target and stay on voice", "标记目标并保持语音"),
        ("do not board until we confirm it is in soft death", "确认软死亡前先别登船"),
        ("clear your crime stat before regrouping", "重新集合前先清犯罪等级"),
        ("refuel and repair before the next bounty mission", "下一单赏金前先补油维修"),
        ("keep escort on the cargo ship until it reaches the station", "货船到站前继续护航"),
        ("tell new players not to confuse the ship name", "提醒新手别把船名认错"),
    ]
    structured_compound_templates = [
        (
            "{event_sentence_en}; {action_en}. {followup_en}",
            "{event_zh}；{action_zh}。{followup_zh}",
        ),
        (
            "{opener_en}{event_en}; also, {secondary_event_en}. {action_sentence_en}.",
            "{opener_zh}{event_zh}；另外，{secondary_event_zh}。{action_zh}。",
        ),
        (
            "If {event_en}, {action_en}; {secondary_event_en}.",
            "如果{event_zh}，{action_zh}；{secondary_event_zh}。",
        ),
        (
            "{opener_en}{event_en}. {secondary_event_sentence_en}; {followup_en}",
            "{opener_zh}{event_zh}。{secondary_event_zh}；{followup_zh}",
        ),
    ]
    multi_ship_templates = [
        (
            "The target at {location_en} is the {left_en}, not the {right_en}.",
            "{location_zh}的目标是{left_zh}，不是{right_zh}。",
        ),
        (
            "The {left_en} at {location_en} is friendly; the {right_en} is the one firing.",
            "{location_zh}那艘{left_zh}是友军，{right_zh}才是在开火的。",
        ),
        (
            "The {left_en} is escorting the {right_en} from {location_en}.",
            "{left_zh}正在从{location_zh}护航{right_zh}。",
        ),
        (
            "There are two ships at {location_en}: the {left_en} is in soft death, and the {right_en} is locking missiles.",
            "{location_zh}有两艘船：{left_zh}软死亡了，{right_zh}在锁导弹。",
        ),
        (
            "I said {left_en}, not {right_en}; do not confuse the ship names in global chat.",
            "我说的是{left_zh}，不是{right_zh}；全局频道里别把船名认错。",
        ),
        (
            "Bring the {left_en} for bounty missions and keep the {right_en} on cargo hauling.",
            "开{left_zh}打赏金，{right_zh}继续跑货。",
        ),
        (
            "If the {left_en} turns red near {location_en}, the {right_en} should not open fire yet.",
            "如果{left_zh}在{location_zh}附近红名，{right_zh}先别开火。",
        ),
        (
            "Voice chat: the {left_en} needs a turret seat, but the {right_en} needs quantum fuel.",
            "语音里说：{left_zh}缺炮塔位，但{right_zh}需要量子燃料。",
        ),
    ]
    multi_ship_slang_templates = [
        (
            "Quick callout: {left_en}, not {right_en}. Check marker.",
            "报点 {left_zh} 不是 {right_zh} 看标记",
        ),
        (
            "Party: the {left_en} is in soft death; the {right_en} is red.",
            "队伍 {left_zh}软死了 {right_zh}红名了",
        ),
        (
            "SC global: take the {left_en} for bounty missions; keep escort on the {right_en}.",
            "全局 开{left_zh}刷赏金 {right_zh}继续护航",
        ),
    ]
    ship_noise_templates = [
        (
            "There is a {ship_en} at {location_en} firing everywhere > F7C-S Hornet Ghost",
            "{location_zh}有个{ship_zh}到处开火 >F7C-S Hornet Ghost",
        ),
        (
            "This feels bad; there is a {ship_en} at {location_en} firing everywhere > F7C-S Hornet Ghost",
            "情况不太对，{location_zh}有个{ship_zh}到处开火 >F7C-S Hornet Ghost",
        ),
        (
            "Global chat says there is a {ship_en} at {location_en} shooting at players @...",
            "全局说{location_zh}有个{ship_zh}在攻击玩家@...",
        ),
        (
            "Quick callout: {location_en} has a {ship_en} spraying fire everywhere [global]",
            "报点 {location_zh}有个{ship_zh}到处乱射[全局]",
        ),
        (
            "Warning: the {ship_en} at {location_en} is shooting players > Drake Cutter",
            "注意 {location_zh}那艘{ship_zh}在攻击玩家>Drake Cutter",
        ),
        (
            "Do not read the trailing tag as the ship; the ship at {location_en} is a {ship_en} > F7C-S Hornet Ghost",
            "别把后面的标签当船名，{location_zh}那艘是{ship_zh} >F7C-S Hornet Ghost",
        ),
    ]
    player_comm_channels = [
        ("", ""),
        ("SC global: ", "全局 "),
        ("Party chat: ", "队伍 "),
        ("Voice: ", "yy里 "),
        ("Need help: ", "来人 "),
        ("New player note: ", "萌新注意 "),
        ("Quick callout: ", "报点 "),
        ("This feels bad; ", "情况不太对 "),
        ("[Org] ", "[Org] "),
        ("[Team] ", "[Team] "),
        ("[Trade] ", "[Trade] "),
    ]
    player_session_channels = player_comm_channels[:-1]
    player_route_channels = [
        ("", ""),
        ("Party chat: ", "队伍 "),
        ("Voice: ", "yy里 "),
        ("Need help: ", "来人 "),
        ("New player note: ", "萌新注意 "),
        ("Quick callout: ", "报点 "),
        ("[Org] ", "[Org] "),
        ("[Team] ", "[Team] "),
    ]
    player_comm_noise_pairs = [
        ("", ""),
        (" > F7C-S Hornet Ghost", " >F7C-S Hornet Ghost"),
        (" @...", "@..."),
        (" [global]", "[全局]"),
        (" [voice]", "[语音]"),
        (" > Drake Cutter", " >Drake Cutter"),
        (" > Aegis Gladius", " >Aegis Gladius"),
        (". Check marker.", " 看标记"),
    ]
    player_comm_state_templates = [
        ("the {ship_en} at {location_en} is firing everywhere", "{location_zh}有个{ship_zh}到处开火"),
        ("the {ship_en} near {location_en} is camping the hangar", "{location_zh}附近有个{ship_zh}在蹲机库"),
        ("the {ship_en} above {location_en} is red", "{location_zh}上空那艘{ship_zh}红名了"),
        ("the {ship_en} at {location_en} is in soft death", "{location_zh}那艘{ship_zh}软死亡了"),
        ("the {ship_en} near {location_en} is locking missiles", "{location_zh}附近那艘{ship_zh}在锁导弹"),
        ("the {ship_en} at {location_en} is full of cargo", "{ship_zh}在{location_zh}满载货物"),
        ("the {ship_en} from {location_en} needs escort", "从{location_zh}出发的{ship_zh}需要护航"),
        ("the {ship_en} at {location_en} is forming a bounty party", "{location_zh}的{ship_zh}在组队打赏金"),
        ("the {ship_en} at {location_en} needs a gunner", "{location_zh}那艘{ship_zh}缺炮手"),
        ("the {ship_en} at {location_en} needs quantum fuel", "{location_zh}那艘{ship_zh}需要量子燃料"),
        ("medical rescue is needed near the {ship_en} at {location_en}", "{location_zh}那艘{ship_zh}附近需要医疗救援"),
        ("the {ship_en} is waiting outside the bunker near {location_en}", "{ship_zh}在{location_zh}附近的地堡外面等人"),
        ("the {ship_en} at {location_en} has a bounty target on board", "{location_zh}那艘{ship_zh}上有赏金目标"),
        ("the {ship_en} claim timer at {location_en} is almost done", "{ship_zh}在{location_zh}的申领时间快好了"),
        ("the {ship_en} near {location_en} is being boarded", "{location_zh}附近那艘{ship_zh}正在被登船"),
        ("the {ship_en} at {location_en} is rubber-banding from desync", "{location_zh}那艘{ship_zh}因为同步很差在瞬移"),
        ("the {ship_en} near {location_en} is mining and needs an escort", "{location_zh}附近那艘{ship_zh}在采矿，需要护航"),
        ("the {ship_en} at {location_en} is doing salvage on a wreck", "{location_zh}那艘{ship_zh}正在打捞残骸"),
        ("the {ship_en} at {location_en} needs repair before it can jump", "{location_zh}那艘{ship_zh}跳跃前需要维修"),
        ("the {ship_en} at {location_en} is waiting on a service beacon", "{location_zh}那艘{ship_zh}在等服务信标"),
        ("the {ship_en} at {location_en} has cargo loose on the cargo grid", "{location_zh}那艘{ship_zh}货物网格上货散了"),
        ("the {ship_en} at {location_en} cannot leave because the hangar doors are stuck", "{location_zh}那艘{ship_zh}因为机库门卡住出不去"),
        ("the {ship_en} crew at {location_en} is stuck at the elevator", "{location_zh}那艘{ship_zh}的船员卡在电梯那里"),
        ("the {ship_en} at {location_en} is trying to share a contract marker", "{location_zh}那艘{ship_zh}想共享合同标记"),
    ]
    player_comm_action_templates = [
        ("please mark it before anyone opens fire", "开火前先标记一下"),
        ("do not shoot until we confirm whether it is friendly", "确认是不是友军前先别开火"),
        ("bring backup and stay on voice", "带支援过来并保持语音"),
        ("ask in global chat if anyone wants to join", "去全局问有没有一起的"),
        ("keep the party away from the pad", "让队伍先离停机坪远点"),
        ("scan it before boarding", "登船前先扫描一下"),
        ("wait for the turret seat to fill before engaging", "炮塔位坐满前先别开打"),
        ("refuel and repair before the next bounty mission", "下一单赏金前先补油维修"),
        ("escort the cargo ship until it reaches the station", "货船到空间站前继续护航"),
        ("drop a medical beacon and call for rescue", "发医疗信标并叫医疗救援"),
        ("clear your crime stat before regrouping", "重新集合前先清犯罪等级"),
        ("pull the bounty target away from the station", "把赏金目标从空间站拉开"),
        ("tell new players not to confuse the ship name", "提醒新手别把船名认错"),
        ("hold fire and check the marker again", "先停火再看一眼标记"),
        ("pick me up if the ship explodes", "船炸了就来接我"),
        ("switch server if the desync gets worse", "同步更差就换服"),
        ("share the contract and wait for everyone to accept it", "共享合同并等所有人接了再走"),
        ("bring a tractor beam and keep the cargo grid clear", "带牵引光束并把货物网格清出来"),
        ("sell the ore after the refinery job finishes", "精炼完成后再去卖矿"),
        ("strip the wreck only after the party marks it", "队伍标记残骸后再开始打捞"),
        ("take the service beacon only if the payment looks right", "服务信标报酬合适再接"),
        ("repair the engines and refill quantum fuel first", "先修引擎并补满量子燃料"),
        ("wait outside the hangar until the elevator bug clears", "在机库外等电梯问题恢复"),
        ("ping the marker again because the party cannot see it", "再 ping 一次标记，队伍看不到"),
    ]
    player_group_tasks = [
        ("a bounty chain", "连打赏金"),
        ("an ERT bounty run", "打ERT赏金"),
        ("cargo hauling", "跑货"),
        ("a salvage loop", "打捞循环"),
        ("mining escort", "采矿护航"),
        ("a bunker clear", "清地堡"),
        ("medical rescue", "医疗救援"),
        ("a service beacon", "服务信标"),
        ("contract sharing", "共享合同"),
        ("an escort route", "护航路线"),
        ("pirate interdiction", "拦海盗"),
        ("a rescue pickup", "救援接送"),
        ("refuel and repair support", "补油维修支援"),
        ("a cargo recovery run", "找回货物"),
    ]
    player_lfg_tags = [
        ("LFG", "LFG"),
        ("LFM", "LFM"),
        ("LF1M", "LF1M"),
        ("crew needed", "缺船员"),
        ("party open", "队伍开放"),
    ]
    player_eta_terms = [
        ("ETA 2 min", "ETA 2分钟"),
        ("ETA 5 min", "ETA 5分钟"),
        ("ready now", "现在就能出发"),
        ("after claim timer", "申领计时结束后"),
        ("after selling cargo", "卖完货以后"),
    ]
    player_crew_roles = [
        ("gunner", "炮手"),
        ("turret gunner", "炮塔手"),
        ("medic", "医疗"),
        ("escort pilot", "护航飞行员"),
        ("cargo runner", "跑货的人"),
        ("scanner", "扫描手"),
        ("salvage operator", "打捞位"),
        ("tractor beam operator", "牵引光束位"),
        ("boarding lead", "登船带队"),
        ("new player who can follow markers", "能跟标记走的萌新"),
    ]
    player_payment_terms = [
        ("payment split after completion", "完成后分账"),
        ("pay on success", "成功后给报酬"),
        ("beacon payout shared", "信标报酬共享"),
        ("cargo profit split", "货款分成"),
        ("tips welcome but not required", "欢迎打赏但不强制"),
        ("we split repair and refuel costs", "补油维修费用平摊"),
    ]
    player_stage_terms = [
        ("before launch", "出发前"),
        ("after refuel and repair", "补油维修后"),
        ("after everyone joins voice", "全员进语音后"),
        ("once the contract is shared", "合同共享后"),
        ("when the marker appears", "标记出来后"),
        ("after the cargo grid is clear", "货物网格清好后"),
        ("before we quantum jump", "量子跳跃前"),
        ("after the beacon is accepted", "信标接了以后"),
    ]
    player_trade_items = [
        ("RMC", "RMC"),
        ("quantanium", "量子矿"),
        ("refined ore", "精炼矿"),
        ("salvage boxes", "打捞箱子"),
        ("cargo crates", "货箱"),
        ("medical supplies", "医疗物资"),
        ("weapons and armor", "武器和护甲"),
        ("ship components", "飞船组件"),
        ("loot boxes", "摸到的箱子"),
        ("recovered cargo", "找回来的货"),
    ]
    player_trade_modes = [
        ("WTB", "WTB"),
        ("WTS", "WTS"),
        ("WTT", "WTT"),
        ("price check", "查价"),
        ("bulk sale", "批量卖"),
    ]
    player_failure_events = [
        ("hit a 30k while loaded", "满货时遇到30k"),
        ("got stuck in the elevator", "卡电梯了"),
        ("cannot open the hangar doors", "机库门打不开"),
        ("lost the party marker", "队伍标记丢了"),
        ("went into soft death", "软死亡了"),
        ("rubber-banded from desync", "因为同步很差来回瞬移"),
        ("lost quantum fuel before the jump", "跳跃前没量子燃料了"),
        ("has loose cargo on the grid", "货物网格上散货了"),
        ("got boarded by pirates", "被海盗登船了"),
        ("has a claim timer longer than expected", "申领时间比预期久"),
    ]
    player_recovery_actions = [
        ("reshare the contract and wait for everyone to accept", "重新共享合同并等所有人接"),
        ("hold fire and scan before boarding", "先停火，登船前扫描"),
        ("ask global chat for pickup", "去全局叫人接送"),
        ("drop a medical beacon and stay on voice", "发医疗信标并保持语音"),
        ("server hop only after we recover the cargo", "找回货物后再换服"),
        ("bring a tractor beam and clear the cargo grid", "带牵引光束并清货物网格"),
        ("refuel and repair before the next route", "下一段路线前先补油维修"),
        ("keep escort outside until the elevator bug clears", "护航在外面等电梯问题恢复"),
    ]
    player_correction_templates = [
        (
            "Correction: I said the {ship_en} at {location_en}, not the {wrong_ship_en}; keep the marker on the ship.",
            "纠正一下：我说的是{location_zh}那艘{ship_zh}，不是{wrong_ship_zh}，标记别换船。",
        ),
        (
            "No, {location_en} is the place and {ship_en} is the ship; do not swap them in chat.",
            "不是，{location_zh}是地点，{ship_zh}是船名，聊天里别把它们互换。",
        ),
        (
            "When I type {term_en}, keep it as the gameplay term; the ship is still the {ship_en}.",
            "我打{term_zh}时保留玩法词，船还是{ship_zh}。",
        ),
        (
            "The quote says {ship_en}; the tag after the message is just noise, not the target.",
            "引用里说的是{ship_zh}，消息后面的标签只是噪声，不是目标。",
        ),
        (
            "Do not turn {term_en} into a vehicle name; the vehicle in this sentence is {ship_en}.",
            "别把{term_zh}转成载具名，这句话里的载具是{ship_zh}。",
        ),
        (
            "If someone asks whether it is the {wrong_ship_en}, answer that the correct ship is the {ship_en}.",
            "如果有人问是不是{wrong_ship_zh}，回答正确的船是{ship_zh}。",
        ),
        (
            "{location_en} is the meetup point, not the ship; the {ship_en} is waiting outside.",
            "{location_zh}是集合点，不是船名；{ship_zh}在外面等。",
        ),
        (
            "This is a correction for new players: {ship_en} is the ship, {term_en} is the gameplay term.",
            "这是给新手的纠正：{ship_zh}是船，{term_zh}是玩法术语。",
        ),
    ]
    player_qa_thread_templates = [
        (
            "[Global] Ren: did you mean the {wrong_ship_en} at {location_en}?\n"
            "[Party] Me: no, I mean the {ship_en}; {term_en} is the gameplay term\n"
            "[Voice] Kai: copy, marking the {ship_en} now",
            "[Global] Ren: 你是说{location_zh}那艘{wrong_ship_zh}吗？\n"
            "[Party] Me: 不是，我说的是{ship_zh}；{term_zh}是玩法术语\n"
            "[Voice] Kai: 收到，现在标记{ship_zh}",
        ),
        (
            "[Trade] Vox: WTB escort for the {ship_en}, or is that {term_en}?\n"
            "[Party] Me: {ship_en} is the ship; {term_en} is the term in that message near {location_en}\n"
            "[Org] Ari: keep both terms exactly",
            "[Trade] Vox: WTB {ship_zh}护航，还是说{term_zh}？\n"
            "[Party] Me: {ship_zh}是船；{term_zh}是{location_zh}那条消息里的术语\n"
            "[Org] Ari: 两个术语都照原意保留",
        ),
        (
            "[Team] Sol: the marker says {location_en}, but global says {ship_en}\n"
            "[Voice] Me: {location_en} is the location; the {ship_en} is parked there while chat talks about {term_en}\n"
            "[Party] Fox: do not translate either one as a random ship",
            "[Team] Sol: 标记写着{location_zh}，但全局说{ship_zh}\n"
            "[Voice] Me: {location_zh}是地点；聊天说{term_zh}时{ship_zh}停在那里\n"
            "[Party] Fox: 两个都别翻成随机船名",
        ),
        (
            "[Global] LFG: need one pilot for {term_en} near {location_en}\n"
            "[Party] Newbie: is {term_en} a ship?\n"
            "[Voice] Me: no, bring the {ship_en}; {term_en} is the gameplay term",
            "[Global] LFG: {location_zh}附近{term_zh}缺一个飞行员\n"
            "[Party] Newbie: {term_zh}是船吗？\n"
            "[Voice] Me: 不是，开{ship_zh}来；{term_zh}是玩法术语",
        ),
    ]
    player_nav_points = [
        ("OM-1", "OM-1"),
        ("OM-3", "OM-3"),
        ("marker 2", "marker 2"),
        ("pad 03", "pad 03"),
        ("hangar 07", "hangar 07"),
        ("ASOP terminal", "ASOP终端"),
        ("comm array", "comm array"),
        ("QT marker", "QT标记"),
        ("party marker", "party marker"),
        ("route marker", "route marker"),
    ]
    player_nav_states = [
        ("holding position", "原地等"),
        ("spooling quantum", "正在量子预热"),
        ("short on quantum fuel", "量子燃料不足"),
        ("blocked by hangar traffic", "被机库交通堵住"),
        ("waiting for ASOP claim", "等ASOP申领"),
        ("missing the party marker", "看不到队伍标记"),
        ("orbiting above the pad", "在停机坪上空盘旋"),
        ("approaching with cargo", "带货接近中"),
        ("breaking off from missile lock", "被锁导弹后脱离"),
        ("waiting for rescue beacon confirmation", "等救援信标确认"),
    ]
    player_nav_actions = [
        ("hold QT until the marker updates", "标记更新前先别QT"),
        ("call the pad number in party chat", "在队伍里报停机坪编号"),
        ("do not translate the nav point as a ship", "别把导航点翻成船名"),
        ("keep the ship name separate from the marker", "把船名和标记分开"),
        ("ask global if the route is camped", "去全局问路线有没有人蹲"),
        ("wait for escort before crossing the route", "等护航到了再过线"),
        ("clear the hangar before bringing cargo in", "带货进来前先清机库"),
        ("share the contract marker again", "再共享一次合同标记"),
    ]
    player_combat_states = [
        ("front shields low", "前盾低"),
        ("rear shields down", "后盾掉了"),
        ("left side taking ballistic fire", "左侧在吃实弹"),
        ("right side taking laser fire", "右侧在吃激光"),
        ("missile lock from above", "上方有missile lock"),
        ("EMP hit the cockpit", "EMP打到驾驶舱"),
        ("distortion damage on weapons", "武器被distortion打了"),
        ("friendly fire on the marker", "标记上有friendly fire"),
        ("ramming attempt near the pad", "停机坪附近有人撞船"),
        ("boarding contact on the ramp", "跳板上有人登船"),
        ("soft death warning", "软死亡警告"),
        ("hostile target is red", "敌对目标红名"),
    ]
    player_system_states = [
        ("weapons offline", "武器离线"),
        ("engines offline", "引擎离线"),
        ("quantum drive offline", "量子驱动离线"),
        ("coolers overheating", "冷却器过热"),
        ("power triangle on weapons", "能量三角给武器"),
        ("power triangle on shields", "能量三角给护盾"),
        ("capacitor empty", "capacitor空了"),
        ("MFD is bugged", "MFD出问题"),
        ("turret seat desynced", "炮塔位同步异常"),
        ("shields recharging", "护盾回充中"),
    ]
    player_combat_actions = [
        ("break lock and flare", "脱锁并放热诱"),
        ("hold fire until the marker is confirmed", "确认标记前先别开火"),
        ("call friendly fire in party chat", "在队伍里报友伤"),
        ("roll left and keep shields forward", "左滚并把前盾顶住"),
        ("power to shields before the next pass", "下一轮前把能量给护盾"),
        ("board only after soft death", "软死亡后再登船"),
        ("scan before firing missiles", "发导弹前先扫描"),
        ("repair engines before chasing", "追击前先修引擎"),
        ("keep the turret on the red target", "炮塔盯红名目标"),
        ("do not confuse the damage callout with a ship name", "别把伤害报点当船名"),
    ]
    player_industrial_jobs = [
        ("quantanium mining", "quantanium采矿"),
        ("ROC mining", "ROC采矿"),
        ("salvage scraping", "刮船打捞"),
        ("RMC hauling", "RMC跑货"),
        ("CM sorting", "CM整理"),
        ("refinery pickup", "精炼提货"),
        ("ore selling", "卖矿"),
        ("cargo grid stacking", "货物网格堆箱"),
        ("wreck stripping", "拆残骸"),
        ("tractor beam loading", "牵引光束装货"),
    ]
    player_industrial_states = [
        ("quantanium timer is running", "quantanium计时开始"),
        ("rock instability is high", "矿石不稳定很高"),
        ("fracture window is tiny", "破裂窗口很窄"),
        ("refinery order is ready", "精炼订单好了"),
        ("cargo grid is full", "货物网格满了"),
        ("RMC boxes are loose", "RMC箱子散了"),
        ("CM crates need sorting", "CM箱子要整理"),
        ("tractor beam is desynced", "牵引光束同步异常"),
        ("sell terminal is bugged", "卖货终端出问题"),
        ("pirates are camping the sale route", "卖货路线上有海盗蹲点"),
        ("scan says the rock is inert", "扫描说矿石惰性"),
        ("refinery job has a long timer", "精炼任务计时很长"),
    ]
    player_industrial_actions = [
        ("mark the rock before anyone starts the laser", "开激光前先标记矿石"),
        ("stop the laser if instability spikes", "不稳定飙升就停激光"),
        ("move boxes before the grid snaps them wrong", "货物网格吸错前先挪箱子"),
        ("keep escort until the cargo reaches the station", "货到空间站前继续护航"),
        ("split the cargo profit after selling", "卖完货后分货款"),
        ("do not translate RMC or CM as ship names", "别把RMC或CM翻成船名"),
        ("bring a tractor beam and clear the ramp", "带牵引光束并清跳板"),
        ("wait for the refinery timer before calling pickup", "等精炼计时结束再叫接送"),
        ("check the sale route before loading the ship", "装船前先看卖货路线"),
        ("keep the ship name separate from the cargo callout", "把船名和货物报点分开"),
    ]
    player_medical_scenarios = [
        ("medical rescue", "医疗救援"),
        ("rescue beacon pickup", "救援信标接人"),
        ("body recovery", "尸体回收"),
        ("crime stat cleanup", "犯罪等级清除"),
        ("security scan", "安保扫描"),
        ("Klescher escape pickup", "克莱舍接人"),
        ("bunker revival", "地堡救人"),
        ("red player check", "红名玩家确认"),
        ("surrender handoff", "投降交接"),
        ("armistice zone extraction", "停火区撤离"),
    ]
    player_medical_states = [
        ("tier 3 injury, still conscious", "T3伤但还醒着"),
        ("incapacitated in the hangar", "倒在机库里"),
        ("medical beacon is up", "医疗信标已经发了"),
        ("rescue beacon payment looks right", "救援信标报酬合适"),
        ("crime stat two after friendly fire", "友伤后犯罪等级2"),
        ("CS3 near security", "安保附近CS3"),
        ("Klescher route is clear", "Klescher路线安全"),
        ("surrender marker is bugged", "投降标记出问题"),
        ("body marker is missing", "尸体标记 body marker丢了"),
        ("med gun is empty", "med gun没药了"),
        ("armistice rules are blocking the pickup", "armistice规则挡住接人"),
        ("security is scanning the wrong ship", "安保在扫错船"),
    ]
    player_medical_actions = [
        ("accept the rescue beacon only after party confirms", "队伍确认后再接救援信标"),
        ("do not shoot the red player until we scan", "扫描前别打红名玩家"),
        ("drag the body to the ramp", "把尸体拖到跳板"),
        ("clear the crime stat before landing", "降落前先清犯罪等级"),
        ("bring a med gun and tractor beam", "带医疗枪和牵引光束"),
        ("keep the ship outside armistice", "让船停在armistice外"),
        ("share the body marker in party chat", "在队伍里共享尸体标记 body marker"),
        ("tell global this is rescue, not bounty", "去全局说明这是救援不是赏金"),
        ("wait for the security scan before opening fire", "等安保扫描后再开火"),
        ("do not confuse Klescher with a ship name", "别把Klescher当船名"),
    ]
    player_ops_scenarios = [
        ("fleet rally", "舰队集结"),
        ("ground team insertion", "地面队突入"),
        ("boarding attempt", "登船尝试"),
        ("distribution center run", "配送中心任务"),
        ("dynamic event wave", "动态事件波次"),
        ("Jumptown pickup", "Jumptown提货"),
        ("XenoThreat escort", "XenoThreat护航"),
        ("bunker clear", "地堡清场"),
        ("airlock breach", "空锁突破"),
        ("evacuation route", "撤离路线"),
        ("sniper overwatch", "狙击手掩护"),
        ("drop ship landing", "登陆艇落地"),
    ]
    player_ops_states = [
        ("ground team is pinned", "地面队被压住"),
        ("drop ship is on final approach", "drop ship正在最后进近"),
        ("airlock is cycling slowly", "airlock开得很慢"),
        ("sniper has eyes on the ramp", "狙击手盯着跳板"),
        ("railgun team is holding the ridge", "railgun队伍守着山脊"),
        ("distribution center elevator is stuck", "配送中心电梯卡住"),
        ("mission marker is under the wrong building", "任务标记在错楼下面"),
        ("party marker and contract marker disagree", "队伍标记和合同标记不一致"),
        ("boarding team is at the hatch", "登船队到舱门了"),
        ("evac point is hot", "撤离点正在交火"),
        ("loot boxes are blocking the corridor", "战利品箱子堵住走廊"),
        ("friendly marker disappeared in the smoke", "友军标记在烟里消失了"),
    ]
    player_ops_actions = [
        ("hold the fleet until the ground team confirms", "等地面队确认后舰队再动"),
        ("land the drop ship nose out", "登陆艇机头朝外落地"),
        ("call the airlock timer in party chat", "在队伍里报airlock计时"),
        ("keep the sniper callout separate from the ship name", "把狙击手报点和船名分开"),
        ("do not translate Jumptown as a location nickname", "别把Jumptown翻成地点外号"),
        ("escort the cargo only after the marker updates", "标记更新后再护送货物"),
        ("clear the corridor before looting", "摸箱子前先清走廊"),
        ("wait for the contract share before pushing", "共享合同后再推进"),
        ("mark the railgun team before the flyover", "低空飞过前先标记railgun队伍"),
        ("extract the team before the next wave", "下一波前先把队伍撤出来"),
    ]
    player_meta_topics = [
        ("server shard check", "服务器分片检查"),
        ("PTU patch test", "PTU补丁测试"),
        ("live build workaround", "正式服绕路办法"),
        ("org event signup", "组织活动报名"),
        ("voice channel setup", "语音频道设置"),
        ("screenshot evidence", "截图取证"),
        ("bug report draft", "bug反馈草稿"),
        ("relog coordination", "重登协调"),
        ("party invite cleanup", "队伍邀请清理"),
        ("hangar instance reset", "机库实例重置"),
        ("claim timer planning", "申领时间安排"),
        ("launcher update wait", "启动器更新等待"),
    ]
    player_meta_states = [
        ("shard feels unstable after the patch", "补丁后shard不太稳定"),
        ("server meshing test is desyncing markers", "服务器网格测试把标记同步乱了"),
        ("launcher is stuck on verifying files", "launcher卡在校验文件"),
        ("PTU build has a known hangar bug", "PTU版本有已知机库bug"),
        ("live build changed the claim timer", "正式服改了claim timer"),
        ("party invite went to the wrong account", "队伍邀请发到错账号"),
        ("voice channel permissions are locked", "语音频道权限锁住了"),
        ("screenshot misses the contract marker", "截图没拍到合同标记"),
        ("crash log is ready to upload", "crash log可以上传了"),
        ("IC report needs reproduction steps", "IC report还缺复现步骤"),
        ("hangar instance ate the ship again", "机库实例又吞船了"),
        ("org roster has two duplicate names", "组织名单有两个重名"),
    ]
    player_meta_actions = [
        ("post the shard ID before everyone relogs", "大家重登前先发shard ID"),
        ("keep the workaround separate from the ship callout", "把绕路办法和船名报点分开"),
        ("paste the patch number with the screenshot", "截图一起贴补丁号"),
        ("move late players to the backup voice channel", "把迟到的人拉到备用语音频道"),
        ("do not translate launcher or PTU as ship names", "别把launcher或PTU翻成船名"),
        ("record the reproduction steps before resetting", "重置前先录复现步骤"),
        ("confirm the org role before sharing the contract", "共享合同前先确认组织权限"),
        ("wait for file verification before joining the server", "等文件校验完再进服务器"),
        ("pin the workaround in party chat", "把绕路办法置顶到队伍聊天"),
        ("attach the crash log after the session", "这局结束后附上crash log"),
    ]
    player_service_topics = [
        ("ship loadout check", "船只整备检查"),
        ("insurance claim", "保险申领"),
        ("expedite timer", "加急申领计时"),
        ("repair and rearm", "维修补弹"),
        ("refuel stop", "补油停靠"),
        ("component swap", "组件更换"),
        ("Vehicle Manager save", "Vehicle Manager保存"),
        ("MobiGlas route check", "MobiGlas路线检查"),
        ("NikNax item lookup", "NikNax物品查询"),
        ("ATC hangar request", "ATC机库请求"),
        ("docking collar alignment", "docking collar对接"),
        ("paint and livery reset", "涂装重置"),
    ]
    player_service_states = [
        ("ASOP terminal shows the wrong ship", "ASOP终端显示错船"),
        ("claim timer is longer than expected", "claim timer比预期长"),
        ("expedite button is greyed out", "expedite按钮灰了"),
        ("repair kiosk charged twice", "维修终端扣了两次钱"),
        ("rearm did not refill missiles", "rearm没有补上missile"),
        ("quantum drive is still stock", "quantum drive还是原厂"),
        ("shield generator is offline", "shield generator离线"),
        ("cooler is overheating after the swap", "cooler换完还过热"),
        ("power plant is missing from inventory", "power plant不在仓库里"),
        ("MobiGlas route will not save", "MobiGlas路线保存不了"),
        ("Vehicle Manager did not apply the loadout", "Vehicle Manager没应用配置"),
        ("ATC assigned the wrong hangar", "ATC分错机库"),
        ("docking collar will not line up", "docking collar对不上"),
        ("paint reset after the claim", "申领后涂装重置了"),
    ]
    player_service_actions = [
        ("screenshot the ASOP page before claiming", "申领前先截图ASOP页面"),
        ("do not confuse the component name with the ship name", "别把组件名当船名"),
        ("wait for the claim timer before expediting", "等claim timer出来再加急"),
        ("save the loadout in Vehicle Manager again", "再去Vehicle Manager保存一次配置"),
        ("check NikNax before buying another component", "买新组件前先查NikNax"),
        ("call ATC only after the party is onboard", "队伍上船后再呼叫ATC"),
        ("repair before rearming if missiles are missing", "导弹没补上就先维修再补弹"),
        ("keep the docking collar callout separate from the ship name", "把docking collar报点和船名分开"),
        ("swap the quantum drive after landing", "落地后再换quantum drive"),
        ("record the paint reset before filing the report", "反馈前先录下涂装重置"),
    ]
    player_mission_topics = [
        ("courier contract", "快递合同"),
        ("investigation mission", "调查任务"),
        ("cave FPS route", "洞穴FPS路线"),
        ("missing person search", "失踪人员搜索"),
        ("box delivery chain", "箱子递送链"),
        ("racing checkpoint", "竞速检查点"),
        ("exploration beacon", "探索信标"),
        ("reputation grind", "声望刷取"),
        ("faction mission split", "阵营任务分工"),
        ("illegal delivery", "非法递送"),
        ("legal salvage contract", "合法打捞合同"),
        ("outpost scanning pass", "前哨扫描航线"),
    ]
    player_mission_states = [
        ("contract marker is under terrain", "合同标记在地形下面"),
        ("delivery locker will not accept the box", "递送柜不收箱子"),
        ("package marker moved to the wrong outpost", "包裹标记跳到错前哨"),
        ("investigation body is missing", "调查目标尸体不见了"),
        ("cave marker points to the wrong tunnel", "洞穴标记指向错洞"),
        ("hostile NPCs respawned behind us", "敌对NPC在身后刷新了"),
        ("checkpoint did not register the lap", "检查点没记录圈速"),
        ("beacon timer is about to expire", "beacon计时快结束"),
        ("reputation payout is delayed", "声望奖励延迟到账"),
        ("faction rep went to the wrong player", "阵营声望给错人"),
        ("illegal cargo is still marked stolen", "非法货物还标着stolen"),
        ("mission objective did not update", "任务目标没有更新"),
    ]
    player_mission_actions = [
        ("share the contract again before entering the cave", "进洞前再共享一次合同"),
        ("do not translate the mission type as a ship name", "别把任务类型翻成船名"),
        ("scan the outpost before dropping the box", "放箱子前先扫描前哨"),
        ("wait for the reputation tick before leaving", "等声望跳了再走"),
        ("call the checkpoint number in party chat", "在队伍里报checkpoint编号"),
        ("mark the body before looting", "摸东西前先标记尸体"),
        ("keep legal and illegal cargo in separate ships", "合法和非法货分开放船"),
        ("record the objective before abandoning the contract", "放弃合同前先录任务目标"),
        ("let the runner take the box while escort covers", "让跑腿拿箱子护航负责掩护"),
        ("use the beacon timer instead of the ship marker", "按beacon计时走不要看船标"),
    ]
    player_gear_topics = [
        ("FPS kit check", "FPS装备检查"),
        ("armor swap", "护甲更换"),
        ("weapon attachment check", "武器配件检查"),
        ("ammo count", "弹药数量"),
        ("medpen restock", "medpen补给"),
        ("tractor tool pickup", "牵引工具拾取"),
        ("multi-tool battery", "multi-tool电池"),
        ("backpack loot split", "背包战利品分配"),
        ("personal inventory cleanup", "个人仓库清理"),
        ("body bag recovery", "尸体背包回收"),
        ("loot crate sweep", "战利品箱清理"),
        ("prison shiv warning", "监狱刀警告"),
    ]
    player_gear_states = [
        ("heavy armor is slowing the runner", "重甲拖慢跑腿"),
        ("undersuit is missing after death", "死后内衬没了"),
        ("helmet is still in local inventory", "头盔还在本地仓库"),
        ("P4-AR has no ammo loaded", "P4-AR没上弹"),
        ("FS-9 magazines are mixed with loot", "FS-9弹匣混在战利品里"),
        ("Coda pistol is on the body marker", "Coda手枪在尸体标记那"),
        ("medpen count is low", "medpen数量不够"),
        ("tractor tool battery is empty", "牵引工具电池空了"),
        ("multi-tool attachment is missing", "multi-tool配件丢了"),
        ("backpack is full of gems", "背包装满gem了"),
        ("loot crate despawn timer is short", "战利品箱despawn计时很短"),
        ("prison shiv is not a ship name", "prison shiv不是船名"),
    ]
    player_gear_actions = [
        ("move ammo to the runner before pushing", "推进前先把弹药给跑腿"),
        ("do not translate the weapon name as a ship", "别把武器名翻成船"),
        ("split medpens before entering the bunker", "进地堡前先分medpen"),
        ("mark the body bag before looting", "摸包前先标记尸体背包"),
        ("drop heavy armor if the route needs sprinting", "路线要冲刺就丢重甲"),
        ("keep local inventory separate from ship inventory", "把本地仓库和船仓库分开"),
        ("check the attachment before buying another gun", "买新枪前先查配件"),
        ("call the loot crate location in party chat", "在队伍里报战利品箱位置"),
        ("save one tractor tool for body recovery", "留一个牵引工具回收尸体"),
        ("record the missing undersuit before relogging", "重登前先录内衬丢失"),
    ]
    player_economy_topics = [
        ("crew payout split", "船员分账"),
        ("beacon payment check", "信标报酬确认"),
        ("escrow trade", "担保交易"),
        ("deposit handoff", "押金交接"),
        ("rental fee", "租船费用"),
        ("refund request", "退款请求"),
        ("service fee", "服务费"),
        ("cargo value estimate", "货值估算"),
        ("bounty payout delay", "赏金到账延迟"),
        ("salvage profit share", "打捞收益分配"),
        ("mining yield split", "采矿收益分配"),
        ("tip transfer", "小费转账"),
    ]
    player_economy_states = [
        ("aUEC transfer is pending", "aUEC转账还在pending"),
        ("UEC balance did not update", "UEC余额没更新"),
        ("beacon payment shows the wrong amount", "信标报酬金额不对"),
        ("escrow holder is not in party", "担保人不在队伍里"),
        ("deposit was paid to the wrong player", "押金付给错人"),
        ("rental fee is higher than agreed", "租船费用比说好的高"),
        ("refund ticket needs a screenshot", "退款单要截图"),
        ("service fee should be split after landing", "服务费落地后再分"),
        ("cargo value changed after the scan", "扫描后货值变了"),
        ("bounty payout is delayed by desync", "赏金报酬因为同步问题延迟"),
        ("profit share is missing the escort cut", "收益分配漏了护航那份"),
        ("tip transfer went through twice", "小费转了两次"),
    ]
    player_economy_actions = [
        ("post the amount before anyone transfers", "转账前先把金额发出来"),
        ("do not translate the payment term as a ship name", "别把付款术语翻成船名"),
        ("split payout only after the cargo is sold", "卖完货后再分账"),
        ("hold the deposit until both sides confirm", "双方确认前先压着押金"),
        ("use party chat for the escrow name", "用队伍聊天确认担保人名字"),
        ("refund the rental fee if the ship claim fails", "申领失败就退租船费用"),
        ("screenshot the beacon payment before accepting", "接信标前先截图信标报酬"),
        ("keep the service fee separate from the cargo value", "把服务费和货值分开算"),
        ("send tip after the rescue is complete", "救援完成后再给小费"),
        ("record the transfer ID before relogging", "重登前先录transfer ID"),
    ]
    player_session_topics = [
        ("party invite", "队伍邀请"),
        ("ready check", "准备确认"),
        ("meetup timing", "集合时间"),
        ("seat assignment", "座位分配"),
        ("voice check", "语音确认"),
        ("server switch", "换服"),
        ("relog wait", "等重登"),
        ("pickup request", "接人请求"),
        ("marker cleanup", "标记整理"),
        ("crew handoff", "换人接手"),
        ("departure call", "出发报点"),
        ("afk notice", "暂离通知"),
    ]
    player_session_states = [
        ("one player did not get the party invite", "还有一个人没收到队伍邀请"),
        ("two people are still on the loading screen", "还有两个人在加载界面"),
        ("the gunner is not in voice yet", "炮手还没进语音"),
        ("the pilot needs one minute before launch", "驾驶员还要一分钟才能出发"),
        ("the marker is on the wrong teammate", "标记挂到错队友身上了"),
        ("the party leader crashed to desktop", "队长闪退到桌面了"),
        ("the ship seat list changed after claim", "申领后座位名单变了"),
        ("someone joined the wrong server", "有人进错服务器了"),
        ("the pickup point changed after the mission share", "共享任务后接人点变了"),
        ("one teammate is afk at the terminal", "有个队友在终端旁边暂离"),
        ("the new player cannot find the hangar", "萌新找不到机库"),
        ("voice is working but party chat is delayed", "语音正常但队伍聊天延迟"),
    ]
    player_session_actions = [
        ("resend the invite before we leave", "出发前再发一次邀请"),
        ("wait until everyone says ready", "等所有人都说准备好了再走"),
        ("put the new player in the copilot seat", "让萌新坐副驾驶位"),
        ("keep the turret seat for the gunner", "炮塔位留给炮手"),
        ("move the marker back to the correct ship", "把标记重新放回正确的船上"),
        ("switch server only after the party regroups", "队伍重新集合后再换服"),
        ("hold outside the hangar until the late player boards", "晚到的人上船前先在机库外等"),
        ("call the pickup point in voice and party chat", "语音和队伍聊天都报一下接人点"),
        ("make the pilot party lead before sharing the contract", "共享合同前先把驾驶员设成队长"),
        ("confirm who is flying before opening the ramp", "开舱门前先确认谁来开船"),
    ]
    player_newbie_topics = [
        ("first flight help", "第一次飞行教学"),
        ("contract pickup", "接合同教学"),
        ("hangar route", "找机库路线"),
        ("party marker follow", "跟队伍标记"),
        ("quantum jump practice", "量子跳跃练习"),
        ("landing request", "降落请求"),
        ("seat and turret lesson", "座位和炮塔教学"),
        ("inventory cleanup", "仓库整理"),
        ("medical rescue lesson", "医疗救援教学"),
        ("crime stat warning", "犯罪等级提醒"),
        ("cargo loading practice", "装货练习"),
        ("claim and retrieve help", "申领取船教学"),
    ]
    player_newbie_states = [
        ("new player cannot find the ASOP terminal", "萌新找不到ASOP终端"),
        ("they accepted the wrong contract", "他接错合同了"),
        ("they are following the wrong marker", "他跟错标记了"),
        ("they do not know which ship is ours", "他不知道哪艘是我们的船"),
        ("they are still inside local inventory", "他还在整理本地仓库"),
        ("they opened fire before we scanned", "还没扫描他就先开火了"),
        ("they cannot find the hangar elevator", "他找不到去机库的电梯"),
        ("they forgot to request landing", "他忘了呼叫降落"),
        ("they left the ship before the pilot parked", "驾驶员停稳前他就下船了"),
        ("they respawned before rescue arrived", "救援到之前他就重生了"),
        ("they put cargo on the wrong grid", "他把货放到错的货物网格上了"),
        ("they claimed the ship instead of retrieving it", "他把取船点成申领了"),
    ]
    player_newbie_actions = [
        ("ask them to follow the party marker first", "先让他跟着队伍标记走"),
        ("share the contract again and wait for acceptance", "重新共享合同并等他接了再走"),
        ("tell them to sit in the copilot seat", "让他先坐副驾驶位"),
        ("mark our ship before opening the ramp", "开舱门前先标一下我们的船"),
        ("walk them to the hangar elevator", "带他走到机库电梯那边"),
        ("do the first quantum jump together", "第一次量子跳跃一起做"),
        ("let the pilot request landing this time", "这次先让驾驶员呼叫降落"),
        ("keep weapons down until the scan finishes", "扫描完成前先别开火"),
        ("wait for medical rescue before respawning", "医疗救援到之前先别重生"),
        ("load one box at a time and check the cargo grid", "一次放一个箱子并检查货物网格"),
        ("show the difference between retrieve and claim", "给他看取船和申领的区别"),
        ("explain the crime stat before anyone shoots", "开火前先讲清楚犯罪等级"),
    ]
    player_dialogue_templates = [
        (
            "[Party] Ari: are you still at {location_en}?\n"
            "[Party] Ren: yes, I am beside the {ship_en}; wait for me before launch.",
            "[Party] Ari: 你还在{location_zh}吗？\n"
            "[Party] Ren: 在，我就在{ship_zh}旁边；出发前等我一下。",
        ),
        (
            "[Voice] Lead: is the {ship_en} ours or a random ship?\n"
            "[Party] Pilot: it is ours; I marked it again.",
            "[Voice] Lead: 这艘{ship_zh}是我们的还是路人的船？\n"
            "[Party] Pilot: 是我们的，我重新标了一下。",
        ),
        (
            "[Global] Newbie: can I join the run at {location_en}?\n"
            "[Party] Lead: yes, board the {ship_en} and stay on voice.",
            "[Global] Newbie: {location_zh}这边能带我一个吗？\n"
            "[Party] Lead: 可以，上{ship_zh}，顺便进语音。",
        ),
        (
            "[Party] Gunner: I see the {other_ship_en}, should I shoot?\n"
            "[Voice] Pilot: no, our target is the {ship_en}; check the marker first.",
            "[Party] Gunner: 我看到{other_ship_zh}了，要开火吗？\n"
            "[Voice] Pilot: 先别，我们目标是{ship_zh}；先看标记。",
        ),
        (
            "[Party] Me: the marker moved at {location_en}; are you still in the {ship_en}?\n"
            "[Party] You: still here, the marker is on the wrong player.",
            "[Party] Me: {location_zh}这边标记跳了，你还在{ship_zh}里吗？\n"
            "[Party] You: 还在，标记挂到错人身上了。",
        ),
        (
            "[Voice] Pilot: who is flying the {ship_en} after refuel?\n"
            "[Party] Crew: you fly, I will take the turret seat.",
            "[Voice] Pilot: {ship_zh}补完油以后谁来开？\n"
            "[Party] Crew: 你开，我去坐炮塔位。",
        ),
        (
            "[Party] Scout: bounty target is leaving {location_en}.\n"
            "[Voice] Lead: pull the {ship_en} out first, then share the contract again.",
            "[Party] Scout: 赏金目标离开{location_zh}了。\n"
            "[Voice] Lead: 先把{ship_zh}开出去，再重新共享合同。",
        ),
        (
            "[Trade] Broker: cargo is ready at {location_en}; is the {ship_en} loaded?\n"
            "[Party] Loader: almost, wait until the cargo grid is clear.",
            "[Trade] Broker: {location_zh}的货准备好了，{ship_zh}装完了吗？\n"
            "[Party] Loader: 快了，等货物网格清好再走。",
        ),
        (
            "[Global] Rescue: who needs pickup near {location_en}?\n"
            "[Party] Medic: the {ship_en} is parked outside; send the beacon.",
            "[Global] Rescue: {location_zh}附近是谁要接人？\n"
            "[Party] Medic: {ship_zh}停在外面，先发信标。",
        ),
        (
            "[Party] Newbie: I clicked claim on the {ship_en}; did I mess up?\n"
            "[Voice] Mentor: it is fine, next time use retrieve if the ship is stored.",
            "[Party] Newbie: 我把{ship_zh}点成申领了，是不是弄错了？\n"
            "[Voice] Mentor: 没事，下次船在仓库里就点取出。",
        ),
        (
            "[Voice] Lead: do not translate {location_en} as the ship name.\n"
            "[Party] Helper: right, {location_en} is the place; the ship is the {ship_en}.",
            "[Voice] Lead: 别把{location_zh}当成船名。\n"
            "[Party] Helper: 对，{location_zh}是地点；船是{ship_zh}。",
        ),
        (
            "[Party] Crew: I lost the party marker near {location_en}.\n"
            "[Voice] Pilot: I am in the {ship_en}; ping me and I will wait.",
            "[Party] Crew: 我在{location_zh}附近看不到队伍标记了。\n"
            "[Voice] Pilot: 我在{ship_zh}里，ping我一下，我等你。",
        ),
    ]
    player_fragment_templates = [
        ("Wait, that is the {ship_en}.", "等下，那是{ship_zh}。"),
        ("Do not shoot the {ship_en}; it is friendly.", "别打{ship_zh}，友军。"),
        ("I am beside the {ship_en} at {location_en}.", "我在{location_zh}的{ship_zh}旁边。"),
        ("Not the {other_ship_en}; I said the {ship_en}.", "不是{other_ship_zh}，我说的是{ship_zh}。"),
        ("I am boarding the {ship_en}; wait a second.", "我在上{ship_zh}，等一下。"),
        ("Marker is wrong; the {ship_en} is over here.", "标记错了，{ship_zh}在这边。"),
        ("The {ship_en} is full; take the next ship.", "{ship_zh}满员了，坐下一艘。"),
        ("I am in the {ship_en}; invite me to party.", "我在{ship_zh}里，拉我进队。"),
        ("That {other_ship_en} is not ours; follow the {ship_en}.", "那艘{other_ship_zh}不是我们的，跟{ship_zh}。"),
        ("Open the ramp on the {ship_en}.", "把{ship_zh}舱门开一下。"),
        ("Hold the {ship_en}; I am still at {location_en}.", "{ship_zh}先别走，我还在{location_zh}。"),
        ("The {ship_en} is the pickup ship, not the target.", "{ship_zh}是接人的船，不是目标。"),
        ("The target is near {location_en}, not inside the {ship_en}.", "目标在{location_zh}附近，不在{ship_zh}里面。"),
        ("Ping the {ship_en} again; I lost the marker.", "再ping一下{ship_zh}，我看不到标记了。"),
    ]
    player_route_topics = [
        ("meetup route", "集合路线"),
        ("pickup route", "接人路线"),
        ("refuel stop", "补油停靠"),
        ("cargo handoff", "交货路线"),
        ("bounty staging", "赏金集合"),
        ("rescue pickup", "救援接人"),
        ("server hop regroup", "换服后集合"),
        ("escort transfer", "护航转场"),
        ("repair detour", "绕路维修"),
        ("departure plan", "出发安排"),
        ("drop-off plan", "下客安排"),
        ("fallback meetup", "备用集合点"),
    ]
    player_route_states = [
        ("destination marker has not been shared yet", "目的地标记还没共享"),
        ("one crew member is still at the old station", "还有一个船员在旧空间站"),
        ("cargo needs to be sold before the jump", "跳走前货要先卖掉"),
        ("quantum fuel is low before the next leg", "下一段前量子燃料不够"),
        ("escort is waiting at the midpoint", "护航在中途点等"),
        ("pickup player is outside armistice", "要接的人在停火区外"),
        ("repair stop will delay the route", "绕去维修会晚一点"),
        ("new player cannot see the route marker", "萌新看不到路线标记"),
        ("target moved away from the original meetup", "目标离开原来的集合点了"),
        ("party marker points to the wrong hangar", "队伍标记指到错机库"),
        ("drop-off point changed after the contract update", "合同更新后下客点变了"),
        ("server feels unstable before departure", "出发前服务器不太稳"),
    ]
    player_route_actions = [
        ("share the destination marker before takeoff", "起飞前先共享目的地标记"),
        ("wait for escort before leaving the station", "离站前等护航到位"),
        ("refuel before taking the cargo route", "跑货路线前先补油"),
        ("pick up the late player before quantum jump", "量子跳跃前先接晚到的人"),
        ("keep the ship outside until everyone boards", "所有人上船前先把船停在外面"),
        ("use the backup meetup if the marker breaks", "标记坏了就用备用集合点"),
        ("sell cargo before switching server", "换服前先把货卖掉"),
        ("call the route in party chat and voice", "队伍聊天和语音都报一下路线"),
        ("keep destination and ship name separate in chat", "聊天里把目的地和船名分开说"),
        ("move the pickup marker after the contract updates", "合同更新后重新放接人标记"),
    ]
    player_decision_topics = [
        ("engage or hold fire", "开火还是先等"),
        ("continue or return to station", "继续任务还是返航"),
        ("sell cargo now or keep hauling", "现在卖货还是继续跑"),
        ("take the service beacon or skip it", "接信标还是跳过"),
        ("repair first or push the route", "先维修还是继续走"),
        ("swap ships or keep the current ship", "换船还是继续用这艘"),
        ("board now or wait for scan", "现在登船还是等扫描"),
        ("server hop or stay together", "换服还是留在本服"),
        ("pick up the late player or leave first", "等晚到的人还是先走"),
        ("split crew or stay on one ship", "分船行动还是同船行动"),
        ("take the bounty chain or change contract", "继续赏金链还是换合同"),
        ("recover cargo or abandon the run", "找回货物还是放弃这趟"),
    ]
    player_decision_states = [
        ("marker is on the wrong ship", "标记挂到错船上了"),
        ("target is close to armistice", "目标离停火区太近"),
        ("cargo value is high enough to avoid risk", "货值够高，不适合冒险"),
        ("one turret seat is still empty", "还有一个炮塔位没人"),
        ("new player is still finding the hangar", "萌新还在找机库"),
        ("quantum fuel is barely enough", "量子燃料刚好够"),
        ("server is starting to stutter", "服务器开始卡了"),
        ("escort has not arrived yet", "护航还没到位"),
        ("ship took engine damage", "船的引擎受损了"),
        ("contract marker changed after sharing", "共享后合同标记变了"),
        ("hostile ship may be friendly", "那艘红名船可能是友军"),
        ("medical beacon is closer than expected", "医疗信标比预期近"),
    ]
    player_decision_actions = [
        ("hold fire until the pilot confirms", "驾驶员确认前先别开火"),
        ("return to the station and repair first", "先回空间站维修"),
        ("sell the cargo before taking another fight", "再打之前先把货卖掉"),
        ("ask party chat for a vote", "在队伍聊天里问一下大家意见"),
        ("keep the ship outside and wait two minutes", "船停在外面再等两分钟"),
        ("swap to the backup ship if the claim timer is short", "申领时间短就换备用船"),
        ("scan before boarding", "登船前先扫描"),
        ("regroup before switching server", "换服前先重新集合"),
        ("pick up the late player before the next jump", "下一跳前先接晚到的人"),
        ("split crew only after voice is clear", "语音确认清楚后再分船"),
        ("change contract if the marker stays wrong", "标记一直不对就换合同"),
        ("abandon the run only after cargo recovery fails", "找不回货再放弃这趟"),
    ]
    player_role_topics = [
        ("pilot handoff", "驾驶交接"),
        ("turret assignment", "炮塔分配"),
        ("copilot task", "副驾驶任务"),
        ("medical seat", "医疗位安排"),
        ("cargo loading", "装货分工"),
        ("scanner duty", "扫描分工"),
        ("escort lead", "护航带队"),
        ("new player guide", "萌新带路"),
        ("boarding lead", "登船指挥"),
        ("salvage operator", "打捞位安排"),
        ("route caller", "路线报点"),
        ("security watch", "警戒分工"),
    ]
    player_role_states = [
        ("assigned player is not on voice yet", "负责的人还没进语音"),
        ("party marker is not clear enough", "队伍标记还不够清楚"),
        ("crew list changed after boarding", "上船后船员名单变了"),
        ("one person is still at the terminal", "还有一个人在终端那边"),
        ("current role holder may need to leave soon", "当前负责的人可能要暂离"),
        ("new player is waiting for instructions", "萌新还在等指挥"),
        ("ship is not ready to launch yet", "船还没准备好出发"),
        ("contract marker moved after sharing", "共享后合同标记跳了"),
        ("voice channel has two people talking over each other", "语音里有两个人同时报点"),
        ("backup crew is on a different ship", "备用船员在另一艘船上"),
        ("station elevator delayed one crew member", "空间站电梯卡住了一个船员"),
        ("target callout is still being confirmed", "目标报点还在确认"),
    ]
    player_role_actions = [
        ("confirm the role in voice before launch", "出发前先在语音里确认岗位"),
        ("write the role assignment in party chat", "把岗位分配发到队伍聊天里"),
        ("keep this role on the same ship until landing", "落地前这个岗位先留在同一艘船上"),
        ("wait for the assigned player before opening the ramp", "等负责的人到位再开舱门"),
        ("move backup crew only after the lead confirms", "带队确认后再调备用船员"),
        ("repeat the callout if the marker changes", "标记变了就重新报一遍"),
        ("do not swap roles during combat", "交火时先别临时换岗位"),
        ("assign one person to explain it to the new player", "安排一个人给萌新解释一下"),
        ("keep the role until the next station stop", "到下一个空间站前先保持这个岗位"),
        ("ask the crew to type ready before launch", "出发前让船员都打一声准备好了"),
        ("confirm the ship name before moving the role", "调整岗位前先确认船名"),
        ("leave the final call to the party lead", "最后决定交给队长"),
    ]
    player_meetup_topics = [
        ("group meetup", "队伍集合"),
        ("late player pickup", "接晚到的人"),
        ("hangar check-in", "机库点名"),
        ("ship boarding order", "登船顺序"),
        ("rally point change", "集合点变更"),
        ("crew ready check", "船员准备确认"),
        ("party marker cleanup", "队伍标记整理"),
        ("voice channel check", "语音频道确认"),
        ("departure countdown", "出发倒计时"),
        ("backup pickup plan", "备用接人方案"),
        ("new player regroup", "萌新重新集合"),
        ("station transfer wait", "空间站转场等人"),
    ]
    player_meetup_states = [
        ("one player loaded into a different shard", "有个队友进到别的分片了"),
        ("two people are still looking for the hangar", "还有两个人在找机库"),
        ("party marker points to the elevator instead of the ship", "队伍标记指到电梯那边了"),
        ("late player has reached the lobby but not the hangar", "晚到的人到大厅了，还没到机库"),
        ("crew cannot hear the pilot clearly in voice", "语音里听不清驾驶员"),
        ("ship is spawned but the ramp is still closed", "船已经刷出来了，但舱门还没开"),
        ("new player followed the wrong marker", "萌新跟错标记了"),
        ("one seat was reserved for the rescue target", "有个座位要留给救援目标"),
        ("party lead is restarting the game", "队长在重启游戏"),
        ("contract marker moved after everyone formed up", "集合完合同标记又跳了"),
        ("someone is still selling cargo at the terminal", "还有人在终端卖货"),
        ("escort pilot is waiting outside the armistice zone", "护航驾驶在停火区外面等"),
    ]
    player_meetup_actions = [
        ("wait at the hangar elevator until everyone checks in", "所有人点名后再离开机库电梯"),
        ("share one marker and remove the old one", "只留一个标记，把旧标记删掉"),
        ("type ready in party chat before boarding", "登船前在队伍聊天里打一声准备好了"),
        ("keep the ship on the pad until the late player arrives", "晚到的人到之前先把船停在平台上"),
        ("move the pickup point outside after takeoff", "起飞后把接人点改到外面"),
        ("repeat the ship name before opening the ramp", "开舱门前再报一遍船名"),
        ("let the new player follow the party lead", "让萌新跟着队长走"),
        ("leave the reserved seat empty until pickup", "接到人之前先空着预留座位"),
        ("restart the ready check after the party lead returns", "队长回来后重新点名"),
        ("do not quantum jump until the marker is fixed", "标记修好前先别量子跳"),
        ("finish selling cargo before calling departure", "卖完货再喊出发"),
        ("tell escort to hold position until boarding is done", "让护航先原地等到登船结束"),
    ]
    player_troubleshoot_topics = [
        ("pre-flight troubleshooting", "出发前排障"),
        ("party sync check", "队伍同步检查"),
        ("ship state check", "船只状态确认"),
        ("marker cleanup", "标记整理"),
        ("relog recovery plan", "重登恢复安排"),
        ("hangar and terminal check", "机库和终端确认"),
        ("cargo recovery check", "货物找回确认"),
        ("crew visibility check", "船员可见性检查"),
        ("route reset check", "路线重置确认"),
        ("inventory refresh check", "仓库刷新确认"),
        ("voice and chat check", "语音和聊天确认"),
        ("departure blocker check", "出发阻塞检查"),
    ]
    player_troubleshoot_states = [
        ("the hangar doors opened for the pilot but not for the crew", "机库门在驾驶员那边开了，但船员这边没开"),
        ("ASOP shows the ship as stored at another station", "ASOP显示船存在另一个空间站"),
        ("party marker follows the old ship instead of the current ship", "队伍标记还跟着旧船，不在当前船上"),
        ("contract marker points to the wrong moon after sharing", "共享后合同标记指到错卫星"),
        ("recovered ship came back without the cargo boxes", "找回的船回来后货箱不见了"),
        ("claim terminal lists the right ship name but wrong hangar", "申领终端船名是对的，但机库不对"),
        ("elevator brings half the crew to a different lobby", "电梯把一半船员送到另一个大厅"),
        ("one player is still loading while the ship is already spawned", "有人还在加载，但船已经刷出来了"),
        ("after relogging the pilot can see the ship but the gunner cannot", "重登后驾驶员能看见船，炮手看不见"),
        ("local inventory updated but ship inventory did not", "本地仓库更新了，但船仓库没同步"),
        ("voice callouts arrive several seconds late", "语音报点会晚几秒才听到"),
        ("quantum route clears itself when the pilot opens the map", "驾驶员打开地图后量子路线会自己消失"),
    ]
    player_troubleshoot_actions = [
        ("keep one person on the ship and have the pilot request the hangar again", "留一个人在船上，让驾驶员重新呼叫机库"),
        ("screenshot the terminal before claiming another ship", "申领别的船之前先截图终端"),
        ("remove the old marker and ping the current ship again", "删掉旧标记，再重新ping当前船"),
        ("share the contract again after everyone accepts the first one", "所有人接完第一遍后再重新共享合同"),
        ("do not store the ship until cargo recovery is confirmed", "确认货物找回前先别存船"),
        ("read the hangar number in party chat before leaving the terminal", "离开终端前把机库号发到队伍聊天"),
        ("wait by the elevator and bring the missing crew member back", "在电梯旁等，把走散的船员带回来"),
        ("hold launch until the loading player types ready", "等加载的人打准备好了再起飞"),
        ("ask the gunner to relog only after the pilot keeps the ship active", "驾驶员保持船在线后再让炮手重登"),
        ("move items only after both inventories refresh", "两个仓库都刷新后再搬东西"),
        ("repeat important callouts in party chat", "重要报点在队伍聊天里再发一遍"),
        ("clear the route and set the destination again before jumping", "清掉路线后重新设目的地再跳"),
    ]

    def sentence_start(text: str) -> str:
        return text[:1].upper() + text[1:] if text else text

    def compact_chat_text(text: str) -> str:
        return (
            text.replace("。", " ")
            .replace("？", " ")
            .replace("！", " ")
            .replace("，", " ")
            .replace("；", " ")
            .replace("：", " ")
            .replace("  ", " ")
            .strip()
        )

    slang_prefixes = [
        ("SC global: ", "全局 "),
        ("Party: ", "队伍 "),
        ("Voice: ", "yy里 "),
        ("Need help: ", "来人 "),
        ("Newbie warning: ", "萌新注意 "),
        ("Quick callout: ", "报点 "),
    ]
    slang_suffixes = [
        (" ASAP.", " 速来"),
        (" Anyone up?", " 有人吗"),
        (" Do not rush.", " 别急着上"),
        (" Check marker.", " 看标记"),
        (" Stay on voice.", " 进语音"),
        (" I need backup.", " 缺支援"),
    ]
    slang_replacements = [
        ("打赏金", "刷赏金"),
        ("软死亡", "软死"),
        ("量子燃料", "量子油"),
        ("犯罪等级", "罪等"),
        ("医疗信标", "救援信标"),
        ("地堡任务", "地堡"),
        ("申领时间", "申领倒计时"),
        ("同步很差", "同步炸了"),
        ("合同标记", "合同点"),
        ("重新共享合同", "重新发合同"),
        ("共享合同", "发合同"),
        ("开火前先标记一下", "开火前先标一下"),
        ("队伍标记", "队伍点"),
    ]

    def slangify_player_chat(text: str) -> str:
        slang_text = compact_chat_text(text)
        for source_phrase, replacement in sorted(
            slang_replacements,
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            slang_text = slang_text.replace(source_phrase, replacement)
        return slang_text

    def pick_other_ship(ship_index: int, salt: int, current_zh: str) -> tuple[str, str]:
        for offset in range(len(ships)):
            candidate_en, candidate_zh = ships[(ship_index + salt + offset) % len(ships)]
            if candidate_zh != current_zh:
                return candidate_en, candidate_zh
        return ships[(ship_index + salt) % len(ships)]

    def pick_other_location(location_index: int, salt: int, current_en: str, current_zh: str) -> tuple[str, str]:
        for offset in range(len(locations)):
            candidate_en, candidate_zh = locations[(location_index + salt + offset) % len(locations)]
            is_same_place = (
                candidate_en == current_en
                or candidate_zh == current_zh
                or candidate_zh in current_zh
                or current_zh in candidate_zh
            )
            if not is_same_place:
                return candidate_en, candidate_zh
        return locations[(location_index + salt) % len(locations)]

    def append_zh_segment(text: str, suffix: str) -> str:
        if not suffix:
            return text
        if not text:
            return suffix.lstrip()
        if re.search(r"[A-Za-z0-9>]$", text) and not suffix.startswith((" ", "\n", "。", "，", "？", "！", "；")):
            return f"{text} {suffix}"
        return f"{text}{suffix}"

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
            for template_index, (en_template, zh_template) in enumerate(gameplay_direct_templates, start=1):
                samples.append(
                    PairSample(
                        key=f"chat_guard:gameplay_direct:{term_index}:{repeat_index + 1}:{template_index}",
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
                for event_index, (event_en_template, event_zh_template) in enumerate(structured_chat_events, start=1):
                    event_en = event_en_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                    )
                    event_zh = event_zh_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                    )
                    opener_index = (repeat_index + location_index + ship_index + event_index) % len(structured_chat_openers)
                    followup_index = (location_index + ship_index + event_index) % len(structured_chat_followups)
                    noise_index = (repeat_index + event_index) % len(structured_noise_pairs)
                    opener_en, opener_zh = structured_chat_openers[opener_index]
                    followup_en, followup_zh = structured_chat_followups[followup_index]
                    noise_en, noise_zh = structured_noise_pairs[noise_index]
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:structured_event:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{event_index}:direct"
                            ),
                            en=f"{opener_en}{event_en}.{followup_en}{noise_en}",
                            zh=f"{opener_zh}{event_zh}。{followup_zh}{noise_zh}",
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    secondary_template_index = (event_index + ship_index + location_index) % len(structured_chat_events)
                    secondary_en_template, secondary_zh_template = structured_chat_events[secondary_template_index]
                    secondary_event_en = secondary_en_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                    )
                    secondary_event_zh = secondary_zh_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                    )
                    action_index = (repeat_index + location_index + ship_index + event_index) % len(
                        structured_compound_actions
                    )
                    action_en_template, action_zh_template = structured_compound_actions[action_index]
                    action_en = action_en_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                    )
                    action_zh = action_zh_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                    )
                    event_sentence_en = sentence_start(event_en)
                    secondary_event_sentence_en = sentence_start(secondary_event_en)
                    action_sentence_en = sentence_start(action_en)
                    compound_template_index = (repeat_index + event_index + ship_index) % len(
                        structured_compound_templates
                    )
                    compound_en_template, compound_zh_template = structured_compound_templates[compound_template_index]
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:structured_compound:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{event_index}:{compound_template_index + 1}"
                            ),
                            en=compound_en_template.format(
                                opener_en=opener_en,
                                opener_zh=opener_zh,
                                event_en=event_en,
                                event_sentence_en=event_sentence_en,
                                event_zh=event_zh,
                                secondary_event_en=secondary_event_en,
                                secondary_event_sentence_en=secondary_event_sentence_en,
                                secondary_event_zh=secondary_event_zh,
                                action_en=action_en,
                                action_sentence_en=action_sentence_en,
                                action_zh=action_zh,
                                followup_en=followup_en.strip(),
                                followup_zh=followup_zh,
                            )
                            + noise_en,
                            zh=compound_zh_template.format(
                                opener_en=opener_en,
                                opener_zh=opener_zh,
                                event_en=event_en,
                                event_sentence_en=event_sentence_en,
                                event_zh=event_zh,
                                secondary_event_en=secondary_event_en,
                                secondary_event_sentence_en=secondary_event_sentence_en,
                                secondary_event_zh=secondary_event_zh,
                                action_en=action_en,
                                action_sentence_en=action_sentence_en,
                                action_zh=action_zh,
                                followup_en=followup_en.strip(),
                                followup_zh=followup_zh,
                            )
                            + noise_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    slang_prefix_en, slang_prefix_zh = slang_prefixes[
                        (repeat_index + event_index + ship_index) % len(slang_prefixes)
                    ]
                    slang_suffix_en, slang_suffix_zh = slang_suffixes[
                        (location_index + event_index + ship_index) % len(slang_suffixes)
                    ]
                    slang_event_zh = compact_chat_text(event_zh)
                    slang_action_zh = compact_chat_text(action_zh)
                    for source_phrase, replacement in sorted(
                        slang_replacements,
                        key=lambda item: len(item[0]),
                        reverse=True,
                    ):
                        slang_event_zh = slang_event_zh.replace(source_phrase, replacement)
                        slang_action_zh = slang_action_zh.replace(source_phrase, replacement)
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:structured_slang:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{event_index}"
                            ),
                            en=f"{slang_prefix_en}{event_sentence_en}; {action_en}.{slang_suffix_en}",
                            zh=f"{slang_prefix_zh}{slang_event_zh} {slang_action_zh}{slang_suffix_zh}",
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    for server_template_index, (server_en_template, server_zh_template) in enumerate(
                        structured_server_events,
                        start=1,
                    ):
                        if (event_index + server_template_index + ship_index) % 3 != 0:
                            continue
                        server_index = (location_index + ship_index + event_index + server_template_index) % len(servers)
                        server_en, server_zh = servers[server_index]
                        samples.append(
                            PairSample(
                                key=(
                                    f"chat_guard:structured_event:{location_index}:{ship_index}:"
                                    f"{repeat_index + 1}:{event_index}:server:{server_template_index}"
                                ),
                                en=server_en_template.format(
                                    server_en=server_en,
                                    server_zh=server_zh,
                                    event_en=event_en,
                                    event_zh=event_zh,
                                )
                                + ".",
                                zh=server_zh_template.format(
                                    server_en=server_en,
                                    server_zh=server_zh,
                                    event_en=event_en,
                                    event_zh=event_zh,
                                )
                                + "。",
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
            for left_index, (left_en, left_zh) in enumerate(ships, start=1):
                right_index = (left_index + location_index + repeat_index) % len(ships)
                right_en, right_zh = ships[right_index]
                if right_en == left_en:
                    right_en, right_zh = ships[(right_index + 1) % len(ships)]
                for template_index, (en_template, zh_template) in enumerate(multi_ship_templates, start=1):
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:multi_ship:{location_index}:{left_index}:"
                                f"{repeat_index + 1}:{template_index}"
                            ),
                            en=en_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                left_en=left_en,
                                left_zh=left_zh,
                                right_en=right_en,
                                right_zh=right_zh,
                            ),
                            zh=zh_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                left_en=left_en,
                                left_zh=left_zh,
                                right_en=right_en,
                                right_zh=right_zh,
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for template_index, (en_template, zh_template) in enumerate(multi_ship_slang_templates, start=1):
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:multi_ship_slang:{location_index}:{left_index}:"
                                f"{repeat_index + 1}:{template_index}"
                            ),
                            en=en_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                left_en=left_en,
                                left_zh=left_zh,
                                right_en=right_en,
                                right_zh=right_zh,
                            ),
                            zh=zh_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                left_en=left_en,
                                left_zh=left_zh,
                                right_en=right_en,
                                right_zh=right_zh,
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
            for ship_index, (ship_en, ship_zh) in enumerate(ships, start=1):
                for template_index, (en_template, zh_template) in enumerate(ship_noise_templates, start=1):
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:ship_noise:{location_index}:{ship_index}:"
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
                for state_index, (state_en_template, state_zh_template) in enumerate(
                    player_comm_state_templates,
                    start=1,
                ):
                    state_en = state_en_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                    )
                    state_zh = state_zh_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                    )
                    for action_index, (action_en_template, action_zh_template) in enumerate(
                        player_comm_action_templates,
                        start=1,
                    ):
                        action_en = action_en_template.format(
                            location_en=location_en,
                            location_zh=location_zh,
                            ship_en=ship_en,
                            ship_zh=ship_zh,
                        )
                        action_zh = action_zh_template.format(
                            location_en=location_en,
                            location_zh=location_zh,
                            ship_en=ship_en,
                            ship_zh=ship_zh,
                        )
                        style_index = (
                            repeat_index + location_index + ship_index + state_index + action_index
                        ) % len(player_comm_channels)
                        noise_index = (
                            repeat_index + (location_index * 3) + ship_index + state_index + action_index
                        ) % len(player_comm_noise_pairs)
                        channel_en, channel_zh = player_comm_channels[style_index]
                        noise_en, noise_zh = player_comm_noise_pairs[noise_index]
                        state_sentence_en = sentence_start(state_en)
                        action_sentence_en = sentence_start(action_en)
                        compact_state_zh = compact_chat_text(state_zh)
                        compact_action_zh = compact_chat_text(action_zh)
                        slang_state_zh = slangify_player_chat(state_zh)
                        slang_action_zh = slangify_player_chat(action_zh)
                        samples.append(
                            PairSample(
                                key=(
                                    f"chat_guard:player_comm_matrix:{location_index}:{ship_index}:"
                                    f"{repeat_index + 1}:{state_index}:{action_index}:standard"
                                ),
                                en=f"{channel_en}{state_sentence_en}. {action_sentence_en}{noise_en}",
                                zh=f"{channel_zh}{state_zh}。{action_zh}{noise_zh}",
                                category="chat",
                                is_priority=True,
                                source="chat_guard",
                            )
                        )
                        samples.append(
                            PairSample(
                                key=(
                                    f"chat_guard:player_comm_matrix:{location_index}:{ship_index}:"
                                    f"{repeat_index + 1}:{state_index}:{action_index}:conditional"
                                ),
                                en=f"{channel_en}If {state_en}, {action_en}{noise_en}",
                                zh=f"{channel_zh}如果{state_zh}，{action_zh}{noise_zh}",
                                category="chat",
                                is_priority=True,
                                source="chat_guard",
                            )
                        )
                        samples.append(
                            PairSample(
                                key=(
                                    f"chat_guard:player_comm_matrix:{location_index}:{ship_index}:"
                                    f"{repeat_index + 1}:{state_index}:{action_index}:compact"
                                ),
                                en=f"{channel_en}{state_sentence_en}; {action_en}{noise_en}",
                                zh=f"{channel_zh}{compact_state_zh} {compact_action_zh}{noise_zh}",
                                category="chat",
                                is_priority=True,
                                source="chat_guard",
                            )
                        )
                        samples.append(
                            PairSample(
                                key=(
                                    f"chat_guard:player_comm_matrix:{location_index}:{ship_index}:"
                                    f"{repeat_index + 1}:{state_index}:{action_index}:slang"
                                ),
                                en=f"{channel_en}{state_sentence_en}; {action_en}{noise_en}",
                                zh=f"{channel_zh}{slang_state_zh} {slang_action_zh}{noise_zh}",
                                category="chat",
                                is_priority=True,
                                source="chat_guard",
                            )
                        )
                for task_index, (task_en, task_zh) in enumerate(player_group_tasks, start=1):
                    tag_en, tag_zh = player_lfg_tags[
                        (repeat_index + location_index + ship_index + task_index) % len(player_lfg_tags)
                    ]
                    eta_en, eta_zh = player_eta_terms[
                        (repeat_index + location_index + task_index) % len(player_eta_terms)
                    ]
                    role_en, role_zh = player_crew_roles[
                        (repeat_index + location_index + ship_index + task_index) % len(player_crew_roles)
                    ]
                    payment_en, payment_zh = player_payment_terms[
                        (repeat_index + ship_index + task_index) % len(player_payment_terms)
                    ]
                    stage_en, stage_zh = player_stage_terms[
                        (repeat_index + location_index + task_index) % len(player_stage_terms)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + location_index + ship_index + task_index + 2) % len(player_comm_channels)
                    ]
                    noise_en, noise_zh = player_comm_noise_pairs[
                        (repeat_index + location_index + ship_index + task_index + 1) % len(player_comm_noise_pairs)
                    ]
                    lfg_en = (
                        f"{channel_en}{tag_en}: {stage_en}, taking the {ship_en} from {location_en} for "
                        f"{task_en}; need one {role_en}; {payment_en}; {eta_en}{noise_en}"
                    )
                    lfg_zh = (
                        f"{channel_zh}{tag_zh}: {stage_zh}从{location_zh}开{ship_zh}{task_zh}，"
                        f"缺{role_zh}，{payment_zh}，{eta_zh}{noise_zh}"
                    )
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_lfg_matrix:{location_index}:{ship_index}:{repeat_index + 1}:{task_index}:standard",
                            en=lfg_en,
                            zh=lfg_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_lfg_matrix:{location_index}:{ship_index}:{repeat_index + 1}:{task_index}:slang",
                            en=lfg_en,
                            zh=slangify_player_chat(lfg_zh),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for trade_index, (item_en, item_zh) in enumerate(player_trade_items, start=1):
                    mode_en, mode_zh = player_trade_modes[
                        (repeat_index + location_index + ship_index + trade_index) % len(player_trade_modes)
                    ]
                    role_en, role_zh = player_crew_roles[
                        (repeat_index + trade_index + ship_index) % len(player_crew_roles)
                    ]
                    payment_en, payment_zh = player_payment_terms[
                        (repeat_index + location_index + trade_index) % len(player_payment_terms)
                    ]
                    trade_en = (
                        f"{mode_en} {item_en} near {location_en}; the {ship_en} is loaded, "
                        f"bring one {role_en}, {payment_en}."
                    )
                    trade_zh = (
                        f"{mode_zh} {location_zh}附近的{item_zh}，{ship_zh}已经装货，"
                        f"带{role_zh}来，{payment_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_trade_matrix:{location_index}:{ship_index}:{repeat_index + 1}:{trade_index}:standard",
                            en=trade_en,
                            zh=trade_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_trade_matrix:{location_index}:{ship_index}:{repeat_index + 1}:{trade_index}:slang",
                            en=trade_en,
                            zh=slangify_player_chat(trade_zh),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for failure_index, (failure_en, failure_zh) in enumerate(player_failure_events, start=1):
                    action_en, action_zh = player_recovery_actions[
                        (repeat_index + location_index + ship_index + failure_index) % len(player_recovery_actions)
                    ]
                    role_en, role_zh = player_crew_roles[
                        (repeat_index + failure_index + location_index) % len(player_crew_roles)
                    ]
                    payment_en, payment_zh = player_payment_terms[
                        (repeat_index + failure_index + ship_index) % len(player_payment_terms)
                    ]
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_recovery_log:{location_index}:{ship_index}:{repeat_index + 1}:{failure_index}",
                            en=(
                                f"[Party] Ari: the {ship_en} at {location_en} {failure_en}\n"
                                f"[Voice] Me: {action_en}\n"
                                f"[Global] LFG: need one {role_en}; {payment_en}"
                            ),
                            zh=(
                                f"[Party] Ari: {location_zh}那艘{ship_zh}{failure_zh}\n"
                                f"[Voice] Me: {action_zh}\n"
                                f"[Global] LFG: 缺{role_zh}，{payment_zh}"
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                wrong_ship_en, wrong_ship_zh = ships[(ship_index + repeat_index + location_index) % len(ships)]
                if wrong_ship_en == ship_en:
                    wrong_ship_en, wrong_ship_zh = ships[(ship_index + repeat_index + location_index + 1) % len(ships)]
                term_en, term_zh = gameplay_terms[
                    (repeat_index + location_index + ship_index) % len(gameplay_terms)
                ]
                for correction_index, (en_template, zh_template) in enumerate(player_correction_templates, start=1):
                    en_text = en_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                        wrong_ship_en=wrong_ship_en,
                        wrong_ship_zh=wrong_ship_zh,
                        term_en=term_en,
                        term_zh=term_zh,
                    )
                    zh_text = zh_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                        wrong_ship_en=wrong_ship_en,
                        wrong_ship_zh=wrong_ship_zh,
                        term_en=term_en,
                        term_zh=term_zh,
                    )
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_correction_matrix:{location_index}:{ship_index}:{repeat_index + 1}:{correction_index}:standard",
                            en=en_text,
                            zh=zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_correction_matrix:{location_index}:{ship_index}:{repeat_index + 1}:{correction_index}:slang",
                            en=en_text,
                            zh=slangify_player_chat(zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for thread_index, (en_template, zh_template) in enumerate(player_qa_thread_templates, start=1):
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_qa_thread:{location_index}:{ship_index}:{repeat_index + 1}:{thread_index}",
                            en=en_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                                wrong_ship_en=wrong_ship_en,
                                wrong_ship_zh=wrong_ship_zh,
                                term_en=term_en,
                                term_zh=term_zh,
                            ),
                            zh=zh_template.format(
                                location_en=location_en,
                                location_zh=location_zh,
                                ship_en=ship_en,
                                ship_zh=ship_zh,
                                wrong_ship_en=wrong_ship_en,
                                wrong_ship_zh=wrong_ship_zh,
                                term_en=term_en,
                                term_zh=term_zh,
                            ),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for nav_index, (nav_en, nav_zh) in enumerate(player_nav_points, start=1):
                    nav_state_en, nav_state_zh = player_nav_states[
                        (repeat_index + location_index + ship_index + nav_index) % len(player_nav_states)
                    ]
                    nav_action_en, nav_action_zh = player_nav_actions[
                        (repeat_index + ship_index + nav_index) % len(player_nav_actions)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + location_index + nav_index) % len(player_comm_channels)
                    ]
                    nav_en_text = (
                        f"{channel_en}{ship_en} at {location_en}, {nav_en}: {nav_state_en}; {nav_action_en}."
                    )
                    nav_zh_text = (
                        f"{channel_zh}{location_zh}{nav_zh}附近的{ship_zh}: {nav_state_zh}；{nav_action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_nav_status:{location_index}:{ship_index}:{repeat_index + 1}:{nav_index}:standard",
                            en=nav_en_text,
                            zh=nav_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_nav_status:{location_index}:{ship_index}:{repeat_index + 1}:{nav_index}:compact",
                            en=nav_en_text,
                            zh=slangify_player_chat(nav_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                nav_en, nav_zh = player_nav_points[(repeat_index + location_index + ship_index) % len(player_nav_points)]
                nav_state_en, nav_state_zh = player_nav_states[
                    (repeat_index + location_index + ship_index + 3) % len(player_nav_states)
                ]
                nav_action_en, nav_action_zh = player_nav_actions[
                    (repeat_index + location_index + ship_index + 5) % len(player_nav_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_nav_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Team] Nav: {location_en} {nav_en}, {ship_en} is {nav_state_en}\n"
                            f"[Party] Me: {nav_action_en}\n"
                            f"[Voice] Kai: ship is {ship_en}, nav point is {nav_en}"
                        ),
                        zh=(
                            f"[Team] Nav: {location_zh} {nav_zh}，{ship_zh}{nav_state_zh}\n"
                            f"[Party] Me: {nav_action_zh}\n"
                            f"[Voice] Kai: 船是{ship_zh}，导航点是{nav_zh}"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for combat_index, (combat_en, combat_zh) in enumerate(player_combat_states, start=1):
                    system_en, system_zh = player_system_states[
                        (repeat_index + location_index + ship_index + combat_index) % len(player_system_states)
                    ]
                    action_en, action_zh = player_combat_actions[
                        (repeat_index + ship_index + combat_index) % len(player_combat_actions)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + combat_index + ship_index) % len(player_comm_channels)
                    ]
                    combat_text_en = (
                        f"{channel_en}{ship_en} at {location_en}: {combat_en}; {system_en}; {action_en}."
                    )
                    combat_text_zh = (
                        f"{channel_zh}{location_zh}那艘{ship_zh}: {combat_zh}；{system_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_combat_status:{location_index}:{ship_index}:{repeat_index + 1}:{combat_index}:standard",
                            en=combat_text_en,
                            zh=combat_text_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=f"chat_guard:player_combat_status:{location_index}:{ship_index}:{repeat_index + 1}:{combat_index}:compact",
                            en=combat_text_en,
                            zh=slangify_player_chat(combat_text_zh),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                combat_en, combat_zh = player_combat_states[
                    (repeat_index + location_index + ship_index + 4) % len(player_combat_states)
                ]
                system_en, system_zh = player_system_states[
                    (repeat_index + location_index + ship_index + 6) % len(player_system_states)
                ]
                action_en, action_zh = player_combat_actions[
                    (repeat_index + location_index + ship_index + 8) % len(player_combat_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_combat_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Voice] Gunner: {ship_en} at {location_en}, {combat_en}\n"
                            f"[Party] Pilot: {system_en}\n"
                            f"[Team] Lead: {action_en}; keep the callout separate from the ship name"
                        ),
                        zh=(
                            f"[Voice] Gunner: {location_zh}那艘{ship_zh}，{combat_zh}\n"
                            f"[Party] Pilot: {system_zh}\n"
                            f"[Team] Lead: {action_zh}；把报点和船名分开"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for industrial_index, (job_en, job_zh) in enumerate(player_industrial_jobs, start=1):
                    state_en, state_zh = player_industrial_states[
                        (repeat_index + location_index + ship_index + industrial_index) % len(player_industrial_states)
                    ]
                    action_en, action_zh = player_industrial_actions[
                        (repeat_index + ship_index + industrial_index) % len(player_industrial_actions)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + industrial_index + location_index) % len(player_comm_channels)
                    ]
                    industrial_en_text = (
                        f"{channel_en}{ship_en} at {location_en} for {job_en}: {state_en}; {action_en}."
                    )
                    industrial_zh_text = (
                        f"{channel_zh}{location_zh}那艘{ship_zh}在做{job_zh}: {state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_industrial_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{industrial_index}:standard"
                            ),
                            en=industrial_en_text,
                            zh=industrial_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_industrial_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{industrial_index}:compact"
                            ),
                            en=industrial_en_text,
                            zh=slangify_player_chat(industrial_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                job_en, job_zh = player_industrial_jobs[
                    (repeat_index + location_index + ship_index) % len(player_industrial_jobs)
                ]
                state_en, state_zh = player_industrial_states[
                    (repeat_index + location_index + ship_index + 4) % len(player_industrial_states)
                ]
                action_en, action_zh = player_industrial_actions[
                    (repeat_index + location_index + ship_index + 6) % len(player_industrial_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_industrial_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Trade] Miner: {job_en} near {location_en}, {state_en}\n"
                            f"[Party] Pilot: the {ship_en} is loaded and waiting\n"
                            f"[Voice] Lead: {action_en}; cargo callout is not the ship name"
                        ),
                        zh=(
                            f"[Trade] Miner: {location_zh}附近{job_zh}，{state_zh}\n"
                            f"[Party] Pilot: {ship_zh}已经装货在等\n"
                            f"[Voice] Lead: {action_zh}；货物报点不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for medical_index, (scenario_en, scenario_zh) in enumerate(player_medical_scenarios, start=1):
                    state_en, state_zh = player_medical_states[
                        (repeat_index + location_index + ship_index + medical_index) % len(player_medical_states)
                    ]
                    action_en, action_zh = player_medical_actions[
                        (repeat_index + ship_index + medical_index) % len(player_medical_actions)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + medical_index + ship_index) % len(player_comm_channels)
                    ]
                    medical_en_text = (
                        f"{channel_en}{scenario_en} near {location_en}: {ship_en} is holding; {state_en}; "
                        f"{action_en}."
                    )
                    medical_zh_text = (
                        f"{channel_zh}{location_zh}附近{scenario_zh}: {ship_zh}先停着；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_medical_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{medical_index}:standard"
                            ),
                            en=medical_en_text,
                            zh=medical_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_medical_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{medical_index}:compact"
                            ),
                            en=medical_en_text,
                            zh=slangify_player_chat(medical_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                scenario_en, scenario_zh = player_medical_scenarios[
                    (repeat_index + location_index + ship_index) % len(player_medical_scenarios)
                ]
                state_en, state_zh = player_medical_states[
                    (repeat_index + location_index + ship_index + 5) % len(player_medical_states)
                ]
                action_en, action_zh = player_medical_actions[
                    (repeat_index + location_index + ship_index + 7) % len(player_medical_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_medical_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Global] Rescue: {scenario_en} near {location_en}, {state_en}\n"
                            f"[Party] Pilot: the {ship_en} is parked outside armistice\n"
                            f"[Voice] Medic: {action_en}; rescue callout is not a ship name"
                        ),
                        zh=(
                            f"[Global] Rescue: {location_zh}附近{scenario_zh}，{state_zh}\n"
                            f"[Party] Pilot: {ship_zh}停在armistice外\n"
                            f"[Voice] Medic: {action_zh}；救援报点不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for ops_index, (scenario_en, scenario_zh) in enumerate(player_ops_scenarios, start=1):
                    state_en, state_zh = player_ops_states[
                        (repeat_index + location_index + ship_index + ops_index) % len(player_ops_states)
                    ]
                    action_en, action_zh = player_ops_actions[
                        (repeat_index + ship_index + ops_index) % len(player_ops_actions)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + ops_index + location_index + ship_index) % len(player_comm_channels)
                    ]
                    ops_en_text = (
                        f"{channel_en}{scenario_en} at {location_en}: {ship_en} is assigned; {state_en}; "
                        f"{action_en}."
                    )
                    ops_zh_text = (
                        f"{channel_zh}{location_zh}{scenario_zh}: {ship_zh}负责；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_ops_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{ops_index}:standard"
                            ),
                            en=ops_en_text,
                            zh=ops_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_ops_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{ops_index}:compact"
                            ),
                            en=ops_en_text,
                            zh=slangify_player_chat(ops_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                scenario_en, scenario_zh = player_ops_scenarios[
                    (repeat_index + location_index + ship_index) % len(player_ops_scenarios)
                ]
                state_en, state_zh = player_ops_states[
                    (repeat_index + location_index + ship_index + 6) % len(player_ops_states)
                ]
                action_en, action_zh = player_ops_actions[
                    (repeat_index + location_index + ship_index + 8) % len(player_ops_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_ops_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Org] Lead: {scenario_en} at {location_en}, {ship_en} is assigned\n"
                            f"[Team] Ground: {state_en}\n"
                            f"[Voice] Flight: {action_en}; operation callout is not the ship name"
                        ),
                        zh=(
                            f"[Org] Lead: {location_zh}{scenario_zh}，{ship_zh}负责\n"
                            f"[Team] Ground: {state_zh}\n"
                            f"[Voice] Flight: {action_zh}；行动报点不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for meta_index, (topic_en, topic_zh) in enumerate(player_meta_topics, start=1):
                    state_en, state_zh = player_meta_states[
                        (repeat_index + location_index + ship_index + meta_index) % len(player_meta_states)
                    ]
                    action_en, action_zh = player_meta_actions[
                        (repeat_index + location_index + meta_index) % len(player_meta_actions)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + meta_index + ship_index) % len(player_comm_channels)
                    ]
                    meta_en_text = (
                        f"{channel_en}{topic_en} near {location_en}: {ship_en} stays assigned; {state_en}; "
                        f"{action_en}."
                    )
                    meta_zh_text = (
                        f"{channel_zh}{location_zh}附近{topic_zh}: {ship_zh}继续负责；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_meta_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{meta_index}:standard"
                            ),
                            en=meta_en_text,
                            zh=meta_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_meta_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{meta_index}:compact"
                            ),
                            en=meta_en_text,
                            zh=slangify_player_chat(meta_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                topic_en, topic_zh = player_meta_topics[
                    (repeat_index + location_index + ship_index) % len(player_meta_topics)
                ]
                state_en, state_zh = player_meta_states[
                    (repeat_index + location_index + ship_index + 6) % len(player_meta_states)
                ]
                action_en, action_zh = player_meta_actions[
                    (repeat_index + location_index + ship_index + 8) % len(player_meta_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_meta_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Local] Helper: {topic_en} near {location_en}, {state_en}\n"
                            f"[Party] Pilot: the {ship_en} stays assigned for now\n"
                            f"[Org] Admin: {action_en}; support callout is not the ship name"
                        ),
                        zh=(
                            f"[Local] Helper: {location_zh}附近{topic_zh}，{state_zh}\n"
                            f"[Party] Pilot: {ship_zh}暂时继续负责\n"
                            f"[Org] Admin: {action_zh}；支持报点不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for service_index, (topic_en, topic_zh) in enumerate(player_service_topics, start=1):
                    state_en, state_zh = player_service_states[
                        (repeat_index + location_index + ship_index + service_index) % len(player_service_states)
                    ]
                    action_en, action_zh = player_service_actions[
                        (repeat_index + ship_index + service_index) % len(player_service_actions)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + service_index + location_index) % len(player_comm_channels)
                    ]
                    service_en_text = (
                        f"{channel_en}{topic_en} at {location_en}: {ship_en} is waiting; {state_en}; "
                        f"{action_en}."
                    )
                    service_zh_text = (
                        f"{channel_zh}{location_zh}{topic_zh}: {ship_zh}在等；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_service_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{service_index}:standard"
                            ),
                            en=service_en_text,
                            zh=service_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_service_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{service_index}:compact"
                            ),
                            en=service_en_text,
                            zh=slangify_player_chat(service_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                topic_en, topic_zh = player_service_topics[
                    (repeat_index + location_index + ship_index) % len(player_service_topics)
                ]
                state_en, state_zh = player_service_states[
                    (repeat_index + location_index + ship_index + 7) % len(player_service_states)
                ]
                action_en, action_zh = player_service_actions[
                    (repeat_index + location_index + ship_index + 9) % len(player_service_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_service_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Local] Mechanic: {topic_en} at {location_en}, {state_en}\n"
                            f"[Party] Pilot: the {ship_en} stays in the hangar\n"
                            f"[Voice] Crew: {action_en}; service callout is not the ship name"
                        ),
                        zh=(
                            f"[Local] Mechanic: {location_zh}{topic_zh}，{state_zh}\n"
                            f"[Party] Pilot: {ship_zh}留在机库\n"
                            f"[Voice] Crew: {action_zh}；整备报点不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for mission_index, (topic_en, topic_zh) in enumerate(player_mission_topics, start=1):
                    state_en, state_zh = player_mission_states[
                        (repeat_index + location_index + ship_index + mission_index) % len(player_mission_states)
                    ]
                    action_en, action_zh = player_mission_actions[
                        (repeat_index + location_index + mission_index) % len(player_mission_actions)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + mission_index + ship_index) % len(player_comm_channels)
                    ]
                    mission_en_text = (
                        f"{channel_en}{topic_en} at {location_en}: {ship_en} is assigned; {state_en}; "
                        f"{action_en}."
                    )
                    mission_zh_text = (
                        f"{channel_zh}{location_zh}{topic_zh}: {ship_zh}负责；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_mission_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{mission_index}:standard"
                            ),
                            en=mission_en_text,
                            zh=mission_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_mission_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{mission_index}:compact"
                            ),
                            en=mission_en_text,
                            zh=slangify_player_chat(mission_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                topic_en, topic_zh = player_mission_topics[
                    (repeat_index + location_index + ship_index) % len(player_mission_topics)
                ]
                state_en, state_zh = player_mission_states[
                    (repeat_index + location_index + ship_index + 7) % len(player_mission_states)
                ]
                action_en, action_zh = player_mission_actions[
                    (repeat_index + location_index + ship_index + 9) % len(player_mission_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_mission_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Party] Runner: {topic_en} at {location_en}, {state_en}\n"
                            f"[Team] Pilot: the {ship_en} is assigned to the mission\n"
                            f"[Voice] Lead: {action_en}; mission callout is not the ship name"
                        ),
                        zh=(
                            f"[Party] Runner: {location_zh}{topic_zh}，{state_zh}\n"
                            f"[Team] Pilot: {ship_zh}负责这个任务\n"
                            f"[Voice] Lead: {action_zh}；任务报点不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for gear_index, (topic_en, topic_zh) in enumerate(player_gear_topics, start=1):
                    state_en, state_zh = player_gear_states[
                        (repeat_index + location_index + ship_index + gear_index) % len(player_gear_states)
                    ]
                    action_en, action_zh = player_gear_actions[
                        (repeat_index + ship_index + gear_index) % len(player_gear_actions)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + gear_index + location_index) % len(player_comm_channels)
                    ]
                    gear_en_text = (
                        f"{channel_en}{topic_en} at {location_en}: {ship_en} is holding; {state_en}; "
                        f"{action_en}."
                    )
                    gear_zh_text = (
                        f"{channel_zh}{location_zh}{topic_zh}: {ship_zh}先等；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_gear_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{gear_index}:standard"
                            ),
                            en=gear_en_text,
                            zh=gear_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_gear_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{gear_index}:compact"
                            ),
                            en=gear_en_text,
                            zh=slangify_player_chat(gear_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                topic_en, topic_zh = player_gear_topics[
                    (repeat_index + location_index + ship_index) % len(player_gear_topics)
                ]
                state_en, state_zh = player_gear_states[
                    (repeat_index + location_index + ship_index + 7) % len(player_gear_states)
                ]
                action_en, action_zh = player_gear_actions[
                    (repeat_index + location_index + ship_index + 9) % len(player_gear_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_gear_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Team] Gear: {topic_en} at {location_en}, {state_en}\n"
                            f"[Party] Pilot: the {ship_en} is holding outside\n"
                            f"[Voice] Runner: {action_en}; gear callout is not the ship name"
                        ),
                        zh=(
                            f"[Team] Gear: {location_zh}{topic_zh}，{state_zh}\n"
                            f"[Party] Pilot: {ship_zh}在外面等\n"
                            f"[Voice] Runner: {action_zh}；装备报点不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for economy_index, (topic_en, topic_zh) in enumerate(player_economy_topics, start=1):
                    state_en, state_zh = player_economy_states[
                        (repeat_index + location_index + ship_index + economy_index) % len(player_economy_states)
                    ]
                    action_en, action_zh = player_economy_actions[
                        (repeat_index + location_index + economy_index) % len(player_economy_actions)
                    ]
                    channel_en, channel_zh = player_comm_channels[
                        (repeat_index + economy_index + ship_index) % len(player_comm_channels)
                    ]
                    economy_en_text = (
                        f"{channel_en}{topic_en} at {location_en}: {ship_en} is waiting; {state_en}; "
                        f"{action_en}."
                    )
                    economy_zh_text = (
                        f"{channel_zh}{location_zh}{topic_zh}: {ship_zh}在等；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_economy_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{economy_index}:standard"
                            ),
                            en=economy_en_text,
                            zh=economy_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_economy_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{economy_index}:compact"
                            ),
                            en=economy_en_text,
                            zh=slangify_player_chat(economy_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                topic_en, topic_zh = player_economy_topics[
                    (repeat_index + location_index + ship_index) % len(player_economy_topics)
                ]
                state_en, state_zh = player_economy_states[
                    (repeat_index + location_index + ship_index + 7) % len(player_economy_states)
                ]
                action_en, action_zh = player_economy_actions[
                    (repeat_index + location_index + ship_index + 9) % len(player_economy_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_economy_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Trade] Broker: {topic_en} at {location_en}, {state_en}\n"
                            f"[Party] Pilot: the {ship_en} is waiting for payout\n"
                            f"[Voice] Lead: {action_en}; payment callout is not the ship name"
                        ),
                        zh=(
                            f"[Trade] Broker: {location_zh}{topic_zh}，{state_zh}\n"
                            f"[Party] Pilot: {ship_zh}在等付款\n"
                            f"[Voice] Lead: {action_zh}；付款报点不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for session_index, (topic_en, topic_zh) in enumerate(player_session_topics, start=1):
                    state_en, state_zh = player_session_states[
                        (repeat_index + location_index + ship_index + session_index) % len(player_session_states)
                    ]
                    action_en, action_zh = player_session_actions[
                        (repeat_index + ship_index + session_index) % len(player_session_actions)
                    ]
                    channel_en, channel_zh = player_session_channels[
                        (repeat_index + session_index + location_index) % len(player_session_channels)
                    ]
                    session_en_text = (
                        f"{channel_en}{topic_en} at {location_en}: {ship_en} is waiting; {state_en}; "
                        f"{action_en}."
                    )
                    session_zh_text = (
                        f"{channel_zh}{location_zh}{topic_zh}: {ship_zh}在等；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_session_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{session_index}:standard"
                            ),
                            en=session_en_text,
                            zh=session_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_session_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{session_index}:compact"
                            ),
                            en=session_en_text,
                            zh=compact_chat_text(session_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                topic_en, topic_zh = player_session_topics[
                    (repeat_index + location_index + ship_index) % len(player_session_topics)
                ]
                state_en, state_zh = player_session_states[
                    (repeat_index + location_index + ship_index + 5) % len(player_session_states)
                ]
                action_en, action_zh = player_session_actions[
                    (repeat_index + location_index + ship_index + 7) % len(player_session_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_session_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Party] Lead: {topic_en} at {location_en}, {state_en}\n"
                            f"[Voice] Pilot: the {ship_en} is holding for the crew\n"
                            f"[Team] Crew: {action_en}; crew chat is not a ship name"
                        ),
                        zh=(
                            f"[Party] Lead: {location_zh}{topic_zh}，{state_zh}\n"
                            f"[Voice] Pilot: {ship_zh}在等船员\n"
                            f"[Team] Crew: {action_zh}；队伍聊天不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for newbie_index, (topic_en, topic_zh) in enumerate(player_newbie_topics, start=1):
                    state_en, state_zh = player_newbie_states[
                        (repeat_index + location_index + ship_index + newbie_index) % len(player_newbie_states)
                    ]
                    action_en, action_zh = player_newbie_actions[
                        (repeat_index + location_index + newbie_index) % len(player_newbie_actions)
                    ]
                    channel_en, channel_zh = player_session_channels[
                        (repeat_index + newbie_index + ship_index) % len(player_session_channels)
                    ]
                    newbie_en_text = (
                        f"{channel_en}{topic_en} at {location_en}: {ship_en} is waiting for the new player; "
                        f"{state_en}; {action_en}."
                    )
                    newbie_zh_text = (
                        f"{channel_zh}{location_zh}{topic_zh}: {ship_zh}在等萌新；"
                        f"{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_newbie_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{newbie_index}:standard"
                            ),
                            en=newbie_en_text,
                            zh=newbie_zh_text,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_newbie_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{newbie_index}:compact"
                            ),
                            en=newbie_en_text,
                            zh=compact_chat_text(newbie_zh_text),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                topic_en, topic_zh = player_newbie_topics[
                    (repeat_index + location_index + ship_index) % len(player_newbie_topics)
                ]
                state_en, state_zh = player_newbie_states[
                    (repeat_index + location_index + ship_index + 4) % len(player_newbie_states)
                ]
                action_en, action_zh = player_newbie_actions[
                    (repeat_index + location_index + ship_index + 6) % len(player_newbie_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_newbie_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Party] Mentor: {topic_en} at {location_en}, {state_en}\n"
                            f"[Voice] Pilot: the {ship_en} is staying parked for now\n"
                            f"[Team] Helper: {action_en}; teaching chat is not a ship name"
                        ),
                        zh=(
                            f"[Party] Mentor: {location_zh}{topic_zh}，{state_zh}\n"
                            f"[Voice] Pilot: {ship_zh}先停着不走\n"
                            f"[Team] Helper: {action_zh}；教学聊天不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for dialogue_index, (en_template, zh_template) in enumerate(player_dialogue_templates, start=1):
                    other_ship_en, other_ship_zh = pick_other_ship(ship_index, dialogue_index, ship_zh)
                    dialogue_en = en_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                        other_ship_en=other_ship_en,
                        other_ship_zh=other_ship_zh,
                    )
                    dialogue_zh = zh_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                        other_ship_en=other_ship_en,
                        other_ship_zh=other_ship_zh,
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_dialogue_thread:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{dialogue_index}:standard"
                            ),
                            en=dialogue_en,
                            zh=dialogue_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_dialogue_thread:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{dialogue_index}:compact"
                            ),
                            en=dialogue_en,
                            zh=compact_chat_text(dialogue_zh),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for fragment_index, (en_template, zh_template) in enumerate(player_fragment_templates, start=1):
                    other_ship_en, other_ship_zh = pick_other_ship(ship_index, fragment_index + 3, ship_zh)
                    fragment_en = en_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                        other_ship_en=other_ship_en,
                        other_ship_zh=other_ship_zh,
                    )
                    fragment_zh = zh_template.format(
                        location_en=location_en,
                        location_zh=location_zh,
                        ship_en=ship_en,
                        ship_zh=ship_zh,
                        other_ship_en=other_ship_en,
                        other_ship_zh=other_ship_zh,
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_fragment:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{fragment_index}:standard"
                            ),
                            en=fragment_en,
                            zh=fragment_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_fragment:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{fragment_index}:plain"
                            ),
                            en=fragment_en,
                            zh=compact_chat_text(fragment_zh),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                for route_index, (topic_en, topic_zh) in enumerate(player_route_topics, start=1):
                    destination_en, destination_zh = pick_other_location(
                        location_index,
                        route_index + ship_index + repeat_index,
                        location_en,
                        location_zh,
                    )
                    state_en, state_zh = player_route_states[
                        (repeat_index + location_index + ship_index + route_index) % len(player_route_states)
                    ]
                    action_en, action_zh = player_route_actions[
                        (repeat_index + ship_index + route_index) % len(player_route_actions)
                    ]
                    channel_en, channel_zh = player_route_channels[
                        (repeat_index + route_index + location_index + ship_index) % len(player_route_channels)
                    ]
                    route_en = (
                        f"{channel_en}{topic_en} from {location_en} to {destination_en}: "
                        f"{ship_en} is assigned; {state_en}; {action_en}."
                    )
                    route_zh = (
                        f"{channel_zh}从{location_zh}到{destination_zh}的{topic_zh}: "
                        f"{ship_zh}负责；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_route_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{route_index}:standard"
                            ),
                            en=route_en,
                            zh=route_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_route_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{route_index}:compact"
                            ),
                            en=route_en,
                            zh=compact_chat_text(route_zh),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                destination_en, destination_zh = pick_other_location(
                    location_index,
                    ship_index + repeat_index + 7,
                    location_en,
                    location_zh,
                )
                route_state_en, route_state_zh = player_route_states[
                    (repeat_index + location_index + ship_index + 3) % len(player_route_states)
                ]
                route_action_en, route_action_zh = player_route_actions[
                    (repeat_index + location_index + ship_index + 5) % len(player_route_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_route_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Party] Nav: route from {location_en} to {destination_en}, {route_state_en}\n"
                            f"[Voice] Pilot: the {ship_en} is assigned for pickup and transfer\n"
                            f"[Team] Crew: {route_action_en}; route callout is not the ship name"
                        ),
                        zh=(
                            f"[Party] Nav: 从{location_zh}到{destination_zh}，{route_state_zh}\n"
                            f"[Voice] Pilot: {ship_zh}负责接人和转场\n"
                            f"[Team] Crew: {route_action_zh}；路线报点不是船名"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for decision_index, (topic_en, topic_zh) in enumerate(player_decision_topics, start=1):
                    state_en, state_zh = player_decision_states[
                        (repeat_index + location_index + ship_index + decision_index) % len(player_decision_states)
                    ]
                    action_en, action_zh = player_decision_actions[
                        (repeat_index + ship_index + decision_index) % len(player_decision_actions)
                    ]
                    other_ship_en, other_ship_zh = pick_other_ship(ship_index, decision_index + 5, ship_zh)
                    channel_en, channel_zh = player_route_channels[
                        (repeat_index + decision_index + ship_index) % len(player_route_channels)
                    ]
                    decision_en = (
                        f"{channel_en}{topic_en} at {location_en}: {ship_en} is assigned, "
                        f"not {other_ship_en}; {state_en}; {action_en}."
                    )
                    decision_zh = (
                        f"{channel_zh}{location_zh}这边{topic_zh}: {ship_zh}负责，不是{other_ship_zh}；"
                        f"{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_decision_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{decision_index}:standard"
                            ),
                            en=decision_en,
                            zh=decision_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_decision_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{decision_index}:compact"
                            ),
                            en=decision_en,
                            zh=compact_chat_text(decision_zh),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                decision_topic_en, decision_topic_zh = player_decision_topics[
                    (repeat_index + location_index + ship_index) % len(player_decision_topics)
                ]
                decision_state_en, decision_state_zh = player_decision_states[
                    (repeat_index + location_index + ship_index + 4) % len(player_decision_states)
                ]
                decision_action_en, decision_action_zh = player_decision_actions[
                    (repeat_index + location_index + ship_index + 6) % len(player_decision_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_decision_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Party] Lead: decision at {location_en}: {decision_topic_en}\n"
                            f"[Voice] Pilot: {ship_en} is assigned, but {decision_state_en}\n"
                            f"[Team] Crew: {decision_action_en}; confirm before changing ships"
                        ),
                        zh=(
                            f"[Party] Lead: {location_zh}这边决定一下：{decision_topic_zh}\n"
                            f"[Voice] Pilot: {ship_zh}负责，但{decision_state_zh}\n"
                            f"[Team] Crew: {decision_action_zh}；换船前先确认"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for role_index, (topic_en, topic_zh) in enumerate(player_role_topics, start=1):
                    state_en, state_zh = player_role_states[
                        (repeat_index + location_index + ship_index + role_index) % len(player_role_states)
                    ]
                    action_en, action_zh = player_role_actions[
                        (repeat_index + ship_index + role_index) % len(player_role_actions)
                    ]
                    other_ship_en, other_ship_zh = pick_other_ship(ship_index, role_index + 9, ship_zh)
                    channel_en, channel_zh = player_route_channels[
                        (repeat_index + role_index + location_index) % len(player_route_channels)
                    ]
                    role_en = (
                        f"{channel_en}{topic_en} on the {ship_en} at {location_en}: "
                        f"{state_en}; {action_en}; do not move this role to the {other_ship_en}."
                    )
                    role_zh = (
                        f"{channel_zh}{location_zh}这边{ship_zh}的{topic_zh}: "
                        f"{state_zh}；{action_zh}；这个岗位别换到{other_ship_zh}上。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_role_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{role_index}:standard"
                            ),
                            en=role_en,
                            zh=role_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_role_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{role_index}:compact"
                            ),
                            en=role_en,
                            zh=compact_chat_text(role_zh),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                role_topic_en, role_topic_zh = player_role_topics[
                    (repeat_index + location_index + ship_index) % len(player_role_topics)
                ]
                role_state_en, role_state_zh = player_role_states[
                    (repeat_index + location_index + ship_index + 4) % len(player_role_states)
                ]
                role_action_en, role_action_zh = player_role_actions[
                    (repeat_index + location_index + ship_index + 6) % len(player_role_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_role_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Party] Lead: {role_topic_en} for the {ship_en} at {location_en}\n"
                            f"[Voice] Crew: {role_state_en}\n"
                            f"[Team] Lead: {role_action_en}; confirm roles before launch"
                        ),
                        zh=(
                            f"[Party] Lead: {location_zh}这边{ship_zh}的{role_topic_zh}\n"
                            f"[Voice] Crew: {role_state_zh}\n"
                            f"[Team] Lead: {role_action_zh}；出发前先确认岗位"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for meetup_index, (topic_en, topic_zh) in enumerate(player_meetup_topics, start=1):
                    state_en, state_zh = player_meetup_states[
                        (repeat_index + location_index + ship_index + meetup_index) % len(player_meetup_states)
                    ]
                    action_en, action_zh = player_meetup_actions[
                        (repeat_index + location_index + meetup_index) % len(player_meetup_actions)
                    ]
                    other_ship_en, other_ship_zh = pick_other_ship(ship_index, meetup_index + 13, ship_zh)
                    channel_en, channel_zh = player_route_channels[
                        (repeat_index + meetup_index + location_index + ship_index) % len(player_route_channels)
                    ]
                    meetup_en = (
                        f"{channel_en}{topic_en} at {location_en}: meet at the {ship_en}, "
                        f"not the {other_ship_en}; {state_en}; {action_en}."
                    )
                    meetup_zh = (
                        f"{channel_zh}{location_zh}这边{topic_zh}: 在{ship_zh}集合，"
                        f"不是{other_ship_zh}；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_meetup_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{meetup_index}:standard"
                            ),
                            en=meetup_en,
                            zh=meetup_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_meetup_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{meetup_index}:compact"
                            ),
                            en=meetup_en,
                            zh=compact_chat_text(meetup_zh),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                meetup_topic_en, meetup_topic_zh = player_meetup_topics[
                    (repeat_index + location_index + ship_index) % len(player_meetup_topics)
                ]
                meetup_state_en, meetup_state_zh = player_meetup_states[
                    (repeat_index + location_index + ship_index + 3) % len(player_meetup_states)
                ]
                meetup_action_en, meetup_action_zh = player_meetup_actions[
                    (repeat_index + location_index + ship_index + 5) % len(player_meetup_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_meetup_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Party] Lead: {meetup_topic_en} at {location_en}; meet at the {ship_en}\n"
                            f"[Voice] Crew: {meetup_state_en}\n"
                            f"[Team] Lead: {meetup_action_en}; confirm before departure"
                        ),
                        zh=(
                            f"[Party] Lead: {location_zh}这边{meetup_topic_zh}，在{ship_zh}集合\n"
                            f"[Voice] Crew: {meetup_state_zh}\n"
                            f"[Team] Lead: {meetup_action_zh}；出发前再确认一次"
                        ),
                        category="chat",
                        is_priority=True,
                        source="chat_guard",
                    )
                )
                for troubleshoot_index, (topic_en, topic_zh) in enumerate(player_troubleshoot_topics, start=1):
                    state_en, state_zh = player_troubleshoot_states[
                        (repeat_index + location_index + ship_index + troubleshoot_index)
                        % len(player_troubleshoot_states)
                    ]
                    action_en, action_zh = player_troubleshoot_actions[
                        (repeat_index + ship_index + troubleshoot_index) % len(player_troubleshoot_actions)
                    ]
                    other_ship_en, other_ship_zh = pick_other_ship(ship_index, troubleshoot_index + 17, ship_zh)
                    channel_en, channel_zh = player_route_channels[
                        (repeat_index + troubleshoot_index + location_index) % len(player_route_channels)
                    ]
                    troubleshoot_en = (
                        f"{channel_en}{topic_en} at {location_en}: keep the {ship_en} active, "
                        f"not the {other_ship_en}; {state_en}; {action_en}."
                    )
                    troubleshoot_zh = (
                        f"{channel_zh}{location_zh}这边{topic_zh}: 先保留{ship_zh}，"
                        f"不是{other_ship_zh}；{state_zh}；{action_zh}。"
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_troubleshoot_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{troubleshoot_index}:standard"
                            ),
                            en=troubleshoot_en,
                            zh=troubleshoot_zh,
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                    samples.append(
                        PairSample(
                            key=(
                                f"chat_guard:player_troubleshoot_status:{location_index}:{ship_index}:"
                                f"{repeat_index + 1}:{troubleshoot_index}:compact"
                            ),
                            en=troubleshoot_en,
                            zh=compact_chat_text(troubleshoot_zh),
                            category="chat",
                            is_priority=True,
                            source="chat_guard",
                        )
                    )
                troubleshoot_topic_en, troubleshoot_topic_zh = player_troubleshoot_topics[
                    (repeat_index + location_index + ship_index) % len(player_troubleshoot_topics)
                ]
                troubleshoot_state_en, troubleshoot_state_zh = player_troubleshoot_states[
                    (repeat_index + location_index + ship_index + 4) % len(player_troubleshoot_states)
                ]
                troubleshoot_action_en, troubleshoot_action_zh = player_troubleshoot_actions[
                    (repeat_index + location_index + ship_index + 6) % len(player_troubleshoot_actions)
                ]
                samples.append(
                    PairSample(
                        key=f"chat_guard:player_troubleshoot_log:{location_index}:{ship_index}:{repeat_index + 1}",
                        en=(
                            f"[Party] Pilot: {troubleshoot_topic_en} at {location_en}; keep the {ship_en} spawned\n"
                            f"[Voice] Crew: {troubleshoot_state_en}\n"
                            f"[Team] Lead: {troubleshoot_action_en}; report the result before departure"
                        ),
                        zh=(
                            f"[Party] Pilot: {location_zh}这边{troubleshoot_topic_zh}，先让{ship_zh}保持刷出状态\n"
                            f"[Voice] Crew: {troubleshoot_state_zh}\n"
                            f"[Team] Lead: {troubleshoot_action_zh}；出发前说一下结果"
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
    base_samples = list(samples)
    for sample_index, sample in enumerate(base_samples, start=1):
        if sample.source != "chat_guard" or sample.category != "chat":
            continue
        if sample_index % 5 == 0:
            wrapper_index = (sample_index // 5 - 1) % len(chat_prefix_wrappers)
            en_prefix, zh_prefix = chat_prefix_wrappers[wrapper_index]
            samples.append(
                PairSample(
                    key=f"{sample.key}:prefix:{wrapper_index + 1}",
                    en=f"{en_prefix}{sample.en}",
                    zh=f"{zh_prefix}{sample.zh}",
                    category="chat",
                    is_priority=True,
                    source="chat_guard",
                )
            )
        if sample_index % 7 == 0:
            wrapper_index = (sample_index // 7 - 1) % len(chat_suffix_wrappers)
            en_suffix, zh_suffix = chat_suffix_wrappers[wrapper_index]
            samples.append(
                PairSample(
                    key=f"{sample.key}:suffix:{wrapper_index + 1}",
                    en=f"{sample.en}{en_suffix}",
                    zh=append_zh_segment(sample.zh, zh_suffix),
                    category="chat",
                    is_priority=True,
                    source="chat_guard",
                )
            )
        if sample_index % 11 == 0:
            wrapper_index = (sample_index // 11 - 1) % len(chat_noise_suffixes)
            en_noise, zh_noise = chat_noise_suffixes[wrapper_index]
            samples.append(
                PairSample(
                    key=f"{sample.key}:noise:{wrapper_index + 1}",
                    en=f"{sample.en}{en_noise}",
                    zh=append_zh_segment(sample.zh, zh_noise),
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
