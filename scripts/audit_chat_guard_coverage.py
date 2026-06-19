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
        "o7",
        "有无",
        "有没有一起的",
        "LF1M",
        "30k",
        "[Org]",
        "[Team]",
        "[Trade]",
        "ETA",
        "OM-3",
        "hangar 07",
        "ASOP",
        "QT",
        "comm array",
        "party marker",
        "route marker",
        "EMP",
        "distortion",
        "MFD",
        "capacitor",
        "missile lock",
        "friendly fire",
        "quantanium",
        "ROC",
        "RMC",
        "CM",
        "refinery",
        "tractor beam",
        "CS3",
        "med gun",
        "body marker",
        "armistice",
        "Klescher",
        "rescue beacon",
        "drop ship",
        "airlock",
        "railgun",
        "Jumptown",
        "XenoThreat",
        "shard ID",
        "server meshing",
        "launcher",
        "PTU",
        "crash log",
        "IC report",
        "Vehicle Manager",
        "MobiGlas",
        "NikNax",
        "ATC",
        "docking collar",
        "quantum drive",
        "shield generator",
        "power plant",
        "expedite",
        "cooler",
        "paint",
        "rearm",
        "checkpoint",
        "beacon",
        "stolen",
        "NPC",
        "P4-AR",
        "FS-9",
        "Coda",
        "medpen",
        "tractor tool",
        "multi-tool",
        "undersuit",
        "local inventory",
        "despawn",
        "aUEC",
        "UEC",
        "escrow",
        "beacon payment",
        "pending",
        "transfer ID",
    ],
    "chat_style": [
        "全局",
        "队伍",
        "语音",
        "来人",
        "萌新注意",
        "报点",
        "情况不太对",
        "完成后分账",
        "成功后给报酬",
        "货款分成",
        "补油维修费用平摊",
        "合同共享后",
        "全员进语音后",
        "缺炮手",
        "缺炮塔手",
        "缺医疗",
        "缺护航飞行员",
        "纠正",
        "不是",
        "我说的是",
        "别翻成",
        "随机船名",
        "保留",
        "引用",
        "地点",
        "玩法术语",
        "消息里的术语",
        "导航点",
        "队伍标记",
        "量子预热",
        "停机坪编号",
        "前盾",
        "后盾",
        "友伤",
        "能量三角",
        "武器离线",
        "犯罪等级",
        "红名",
        "医疗信标",
        "救援信标",
        "尸体标记",
        "停火区",
        "投降标记",
        "安保",
        "医疗救援",
        "救援报点",
        "舰队",
        "地面队",
        "登陆艇",
        "空锁",
        "狙击手",
        "配送中心",
        "登船",
        "撤离点",
        "行动报点",
        "服务器分片",
        "正式服",
        "补丁",
        "组织活动",
        "语音频道",
        "截图",
        "复现步骤",
        "重登",
        "支持报点",
        "船只整备",
        "保险申领",
        "加急申领",
        "维修补弹",
        "组件",
        "配置",
        "整备报点",
        "涂装",
        "快递合同",
        "调查任务",
        "洞穴",
        "失踪人员",
        "声望",
        "阵营",
        "非法递送",
        "合法打捞",
        "任务报点",
        "FPS装备",
        "护甲",
        "武器配件",
        "弹药",
        "背包",
        "战利品",
        "个人仓库",
        "尸体背包",
        "装备报点",
        "分账",
        "押金",
        "租船",
        "退款",
        "服务费",
        "货值",
        "收益分配",
        "付款报点",
        "队伍邀请",
        "准备确认",
        "集合时间",
        "座位分配",
        "语音确认",
        "换服",
        "等重登",
        "接人请求",
        "标记整理",
        "换人接手",
        "出发报点",
        "暂离通知",
        "副驾驶位",
        "队伍聊天",
        "第一次飞行教学",
        "接合同教学",
        "找机库路线",
        "跟队伍标记",
        "量子跳跃练习",
        "降落请求",
        "座位和炮塔教学",
        "仓库整理",
        "医疗救援教学",
        "犯罪等级提醒",
        "装货练习",
        "申领取船教学",
        "教学聊天",
        "能带我一个吗",
        "我们的船",
        "路人的船",
        "先别",
        "目标是",
        "重新共享合同",
        "装完了吗",
        "先发信标",
        "看不到队伍标记",
        "地点",
        "等下",
        "别打",
        "友军",
        "标记错了",
        "拉我进队",
        "不是我们的",
        "舱门开一下",
        "满员了",
        "先别走",
        "接人的船",
        "集合路线",
        "接人路线",
        "补油停靠",
        "交货路线",
        "赏金集合",
        "救援接人",
        "换服后集合",
        "护航转场",
        "绕路维修",
        "出发安排",
        "下客安排",
        "备用集合点",
        "目的地标记",
        "路线标记",
        "路线报点",
        "接人标记",
        "开火还是先等",
        "继续任务还是返航",
        "现在卖货还是继续跑",
        "接信标还是跳过",
        "先维修还是继续走",
        "换船还是继续用这艘",
        "现在登船还是等扫描",
        "换服还是留在本服",
        "等晚到的人还是先走",
        "分船行动还是同船行动",
        "继续赏金链还是换合同",
        "找回货物还是放弃这趟",
        "队伍聊天里问一下大家意见",
        "换船前先确认",
        "驾驶交接",
        "炮塔分配",
        "副驾驶任务",
        "医疗位安排",
        "装货分工",
        "扫描分工",
        "护航带队",
        "萌新带路",
        "登船指挥",
        "打捞位安排",
        "路线报点",
        "警戒分工",
        "出发前先确认岗位",
        "队伍集合",
        "接晚到的人",
        "机库点名",
        "登船顺序",
        "集合点变更",
        "船员准备确认",
        "队伍标记整理",
        "语音频道确认",
        "出发倒计时",
        "备用接人方案",
        "萌新重新集合",
        "空间站转场等人",
        "出发前再确认一次",
        "出发前排障",
        "队伍同步检查",
        "船只状态确认",
        "机库和终端确认",
        "货物找回确认",
        "出发阻塞检查",
        "机库门在驾驶员那边开了",
        "ASOP显示船存在另一个空间站",
        "队伍标记还跟着旧船",
        "共享后合同标记指到错卫星",
        "找回的船回来后货箱不见了",
        "重登后驾驶员能看见船",
        "出发前说一下结果",
        "安保协调",
        "红名状态确认",
        "停火区行动确认",
        "目标身份确认",
        "非法状态处理",
        "赏金和队伍标记确认",
        "禁区行动协调",
        "投降或撤离决定",
        "罚金和监狱安排",
        "Klescher接人协调",
        "安检前准备",
        "安保反应协调",
        "动手前先确认",
        "船员权限协调",
        "共享船权限确认",
        "队伍权限确认",
        "登船权限确认",
        "货舱和座位权限确认",
        "医疗和炮塔权限确认",
        "队长转移确认",
        "机库和对接权限确认",
        "仓库访问确认",
        "好友和队伍邀请确认",
        "组织船员权限确认",
        "起飞前权限确认",
        "起飞前确认权限",
        "活动时间协调",
        "船员在线确认",
        "晚到接手安排",
        "提前下线交接",
        "替补船员安排",
        "短时间活动安排",
        "长线任务时间确认",
        "中途休息确认",
        "重启前时间确认",
        "任务后交接",
        "集合时间确认",
        "备用驾驶时间确认",
        "出发前确认时间",
        "进出港协调",
        "ATC和机库确认",
        "进场路线协调",
        "交通顺序协调",
        "停机坪和对接协调",
        "空间站空域协调",
        "带货进场协调",
        "接人进场协调",
        "舰队移动协调",
        "最后进场确认",
        "复飞后重新进场",
        "机库号确认",
        "最后进场前确认",
        "引擎离线",
        "报点和船名",
        "量子矿",
        "矿石不稳定",
        "破裂窗口",
        "精炼订单",
        "货物报点",
        "货物网格",
        "卖货路线",
        "卖货终端",
        "拆残骸",
    ],
}

UNNATURAL_MIXED_SOURCE_PATTERNS = [
    "补油repair",
    "先marker",
    "为desync",
    "带tractor beam",
    "把cargo grid",
    "等elevator",
    "hangar door卡",
    "bounty目标",
    "med beacon",
    "med rescue",
    "去sell ore",
    "fillq油",
    "要q油",
    "markerwreck",
    "salvagewreck",
    "share contract",
    "service beacon报",
    "refinery完成",
    "打bounty",
    "跑cargo",
    "继续escort",
    "soft death了",
    " red了",
    "turret位",
    "锁missile",
    "蹲hangar",
    "需要repair",
    "cargo grid上",
    "换route",
    "别landing",
    "marker 2在",
    "跟marker",
    "喊escort",
    "plz",
    "ASAP",
    "WTB",
    "WTS",
    "WTT",
    "yy ",
    "yy里",
    "elevator bug",
]
NOISE_TAG_GLUE_RE = re.compile(r">(?:F7C-S Hornet Ghost|Aegis Gladius|Drake Cutter)[\u3400-\u9fff]")


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


def find_unnatural_mixed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for row in rows:
        source = row.get("source", "")
        hits = [pattern for pattern in UNNATURAL_MIXED_SOURCE_PATTERNS if pattern in source]
        if NOISE_TAG_GLUE_RE.search(source):
            hits.append("noise_tag_glue")
        if hits:
            matches.append(
                {
                    "key": row.get("key", ""),
                    "hits": hits,
                    "source": source,
                }
            )
    return matches


def build_report(rows: list[dict[str, Any]], aliases_file: Path, terms_file: Path) -> dict[str, Any]:
    alias_pairs = load_alias_pairs(aliases_file)
    vehicle_pairs = load_term_pairs(terms_file, category_filter="vehicle")
    location_pairs = load_term_pairs(terms_file, category_filter="location")
    gameplay_pairs = load_term_pairs(terms_file, category_filter="gameplay")
    root_counts = Counter(key_root(row.get("key", "")) for row in rows)
    chat_counts = Counter(chat_subkey(row.get("key", "")) for row in rows if row.get("key", "").startswith("chat_guard:"))
    player_lfg_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_lfg_matrix:")]
    player_trade_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_trade_matrix:")]
    player_recovery_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_recovery_log:")]
    player_correction_rows = [
        row for row in rows if row.get("key", "").startswith("chat_guard:player_correction_matrix:")
    ]
    player_qa_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_qa_thread:")]
    player_nav_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_nav_status:")]
    player_nav_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_nav_log:")]
    player_combat_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_combat_status:")]
    player_combat_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_combat_log:")]
    player_industrial_rows = [
        row for row in rows if row.get("key", "").startswith("chat_guard:player_industrial_status:")
    ]
    player_industrial_log_rows = [
        row for row in rows if row.get("key", "").startswith("chat_guard:player_industrial_log:")
    ]
    player_medical_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_medical_status:")]
    player_medical_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_medical_log:")]
    player_ops_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_ops_status:")]
    player_ops_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_ops_log:")]
    player_meta_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_meta_status:")]
    player_meta_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_meta_log:")]
    player_service_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_service_status:")]
    player_service_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_service_log:")]
    player_mission_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_mission_status:")]
    player_mission_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_mission_log:")]
    player_gear_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_gear_status:")]
    player_gear_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_gear_log:")]
    player_economy_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_economy_status:")]
    player_economy_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_economy_log:")]
    player_session_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_session_status:")]
    player_session_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_session_log:")]
    player_newbie_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_newbie_status:")]
    player_newbie_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_newbie_log:")]
    player_dialogue_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_dialogue_thread:")]
    player_fragment_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_fragment:")]
    player_route_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_route_status:")]
    player_route_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_route_log:")]
    player_decision_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_decision_status:")]
    player_decision_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_decision_log:")]
    player_role_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_role_status:")]
    player_role_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_role_log:")]
    player_meetup_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_meetup_status:")]
    player_meetup_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_meetup_log:")]
    player_troubleshoot_rows = [
        row for row in rows if row.get("key", "").startswith("chat_guard:player_troubleshoot_status:")
    ]
    player_troubleshoot_log_rows = [
        row for row in rows if row.get("key", "").startswith("chat_guard:player_troubleshoot_log:")
    ]
    player_security_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_security_status:")]
    player_security_log_rows = [
        row for row in rows if row.get("key", "").startswith("chat_guard:player_security_log:")
    ]
    player_access_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_access_status:")]
    player_access_log_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_access_log:")]
    player_schedule_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_schedule_status:")]
    player_schedule_log_rows = [
        row for row in rows if row.get("key", "").startswith("chat_guard:player_schedule_log:")
    ]
    player_landing_rows = [row for row in rows if row.get("key", "").startswith("chat_guard:player_landing_status:")]
    player_landing_log_rows = [
        row for row in rows if row.get("key", "").startswith("chat_guard:player_landing_log:")
    ]
    target_cjk = [row for row in rows if re.search(r"[\u3400-\u9fff]", row.get("target", ""))]
    unnatural_mixed_rows = find_unnatural_mixed_rows(rows)
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
        "unnatural_mixed_source_count": len(unnatural_mixed_rows),
        "unnatural_mixed_source_examples": unnatural_mixed_rows[:20],
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
        "player_lfg_rows": len(player_lfg_rows),
        "player_trade_rows": len(player_trade_rows),
        "player_recovery_rows": len(player_recovery_rows),
        "player_correction_rows": len(player_correction_rows),
        "player_qa_rows": len(player_qa_rows),
        "player_nav_rows": len(player_nav_rows),
        "player_nav_log_rows": len(player_nav_log_rows),
        "player_combat_rows": len(player_combat_rows),
        "player_combat_log_rows": len(player_combat_log_rows),
        "player_industrial_rows": len(player_industrial_rows),
        "player_industrial_log_rows": len(player_industrial_log_rows),
        "player_medical_rows": len(player_medical_rows),
        "player_medical_log_rows": len(player_medical_log_rows),
        "player_ops_rows": len(player_ops_rows),
        "player_ops_log_rows": len(player_ops_log_rows),
        "player_meta_rows": len(player_meta_rows),
        "player_meta_log_rows": len(player_meta_log_rows),
        "player_service_rows": len(player_service_rows),
        "player_service_log_rows": len(player_service_log_rows),
        "player_mission_rows": len(player_mission_rows),
        "player_mission_log_rows": len(player_mission_log_rows),
        "player_gear_rows": len(player_gear_rows),
        "player_gear_log_rows": len(player_gear_log_rows),
        "player_economy_rows": len(player_economy_rows),
        "player_economy_log_rows": len(player_economy_log_rows),
        "player_session_rows": len(player_session_rows),
        "player_session_log_rows": len(player_session_log_rows),
        "player_newbie_rows": len(player_newbie_rows),
        "player_newbie_log_rows": len(player_newbie_log_rows),
        "player_dialogue_rows": len(player_dialogue_rows),
        "player_fragment_rows": len(player_fragment_rows),
        "player_route_rows": len(player_route_rows),
        "player_route_log_rows": len(player_route_log_rows),
        "player_decision_rows": len(player_decision_rows),
        "player_decision_log_rows": len(player_decision_log_rows),
        "player_role_rows": len(player_role_rows),
        "player_role_log_rows": len(player_role_log_rows),
        "player_meetup_rows": len(player_meetup_rows),
        "player_meetup_log_rows": len(player_meetup_log_rows),
        "player_troubleshoot_rows": len(player_troubleshoot_rows),
        "player_troubleshoot_log_rows": len(player_troubleshoot_log_rows),
        "player_security_rows": len(player_security_rows),
        "player_security_log_rows": len(player_security_log_rows),
        "player_access_rows": len(player_access_rows),
        "player_access_log_rows": len(player_access_log_rows),
        "player_schedule_rows": len(player_schedule_rows),
        "player_schedule_log_rows": len(player_schedule_log_rows),
        "player_landing_rows": len(player_landing_rows),
        "player_landing_log_rows": len(player_landing_log_rows),
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
    print(f"unnatural_mixed_source_count: {report['unnatural_mixed_source_count']}")
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
    print(f"player_lfg_rows: {report['player_lfg_rows']}")
    print(f"player_trade_rows: {report['player_trade_rows']}")
    print(f"player_recovery_rows: {report['player_recovery_rows']}")
    print(f"player_correction_rows: {report['player_correction_rows']}")
    print(f"player_qa_rows: {report['player_qa_rows']}")
    print(f"player_nav_rows: {report['player_nav_rows']}")
    print(f"player_nav_log_rows: {report['player_nav_log_rows']}")
    print(f"player_combat_rows: {report['player_combat_rows']}")
    print(f"player_combat_log_rows: {report['player_combat_log_rows']}")
    print(f"player_industrial_rows: {report['player_industrial_rows']}")
    print(f"player_industrial_log_rows: {report['player_industrial_log_rows']}")
    print(f"player_medical_rows: {report['player_medical_rows']}")
    print(f"player_medical_log_rows: {report['player_medical_log_rows']}")
    print(f"player_ops_rows: {report['player_ops_rows']}")
    print(f"player_ops_log_rows: {report['player_ops_log_rows']}")
    print(f"player_meta_rows: {report['player_meta_rows']}")
    print(f"player_meta_log_rows: {report['player_meta_log_rows']}")
    print(f"player_service_rows: {report['player_service_rows']}")
    print(f"player_service_log_rows: {report['player_service_log_rows']}")
    print(f"player_mission_rows: {report['player_mission_rows']}")
    print(f"player_mission_log_rows: {report['player_mission_log_rows']}")
    print(f"player_gear_rows: {report['player_gear_rows']}")
    print(f"player_gear_log_rows: {report['player_gear_log_rows']}")
    print(f"player_economy_rows: {report['player_economy_rows']}")
    print(f"player_economy_log_rows: {report['player_economy_log_rows']}")
    print(f"player_session_rows: {report['player_session_rows']}")
    print(f"player_session_log_rows: {report['player_session_log_rows']}")
    print(f"player_newbie_rows: {report['player_newbie_rows']}")
    print(f"player_newbie_log_rows: {report['player_newbie_log_rows']}")
    print(f"player_dialogue_rows: {report['player_dialogue_rows']}")
    print(f"player_fragment_rows: {report['player_fragment_rows']}")
    print(f"player_route_rows: {report['player_route_rows']}")
    print(f"player_route_log_rows: {report['player_route_log_rows']}")
    print(f"player_decision_rows: {report['player_decision_rows']}")
    print(f"player_decision_log_rows: {report['player_decision_log_rows']}")
    print(f"player_role_rows: {report['player_role_rows']}")
    print(f"player_role_log_rows: {report['player_role_log_rows']}")
    print(f"player_meetup_rows: {report['player_meetup_rows']}")
    print(f"player_meetup_log_rows: {report['player_meetup_log_rows']}")
    print(f"player_troubleshoot_rows: {report['player_troubleshoot_rows']}")
    print(f"player_troubleshoot_log_rows: {report['player_troubleshoot_log_rows']}")
    print(f"player_security_rows: {report['player_security_rows']}")
    print(f"player_security_log_rows: {report['player_security_log_rows']}")
    print(f"player_access_rows: {report['player_access_rows']}")
    print(f"player_access_log_rows: {report['player_access_log_rows']}")
    print(f"player_schedule_rows: {report['player_schedule_rows']}")
    print(f"player_schedule_log_rows: {report['player_schedule_log_rows']}")
    print(f"player_landing_rows: {report['player_landing_rows']}")
    print(f"player_landing_log_rows: {report['player_landing_log_rows']}")
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
    if report["unnatural_mixed_source_examples"]:
        print("unnatural_mixed_source_examples:")
        for example in report["unnatural_mixed_source_examples"]:
            print(f"  {example['key']}: {example['hits']} :: {example['source'][:180]}")

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
