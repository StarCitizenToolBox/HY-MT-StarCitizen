import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryResult:
    category: str
    is_priority: bool


class StarCitizenKeyClassifier:
    """Key classifier adapted from Opus-MT-StarCitizen's priority categories."""

    PRIORITY_CATEGORIES: dict[str, list[str]] = {
        "location": [
            r"^Bacchus(?!.*_Desc).*",
            r"^Cano(?!.*_Desc).*",
            r"^Castra(?!.*_Desc).*",
            r"^Delamar(?!.*_Desc).*",
            r"^Ellis(?!.*_Desc).*",
            r"^Goss(?!.*_Desc).*",
            r"^Hadrian(?!.*_Desc).*",
            r"^Levski_Shop_Teach(?!.*_Desc).*",
            r"^Magnus(?!.*_Desc).*",
            r"^Nyx(?!.*_Desc).*",
            r"^Oso(?!.*_Desc).*",
            r"^Pyro(?!.*_Desc).*",
            r"^Stanton(?!.*_Desc).*",
            r"^Taranis(?!.*_Desc).*",
            r"^Tarpits(?!.*_Desc).*",
            r"^Tayac(?!.*_Desc).*",
            r"^Terra(?!.*_Desc).*",
            r"^Virgil(?!.*_Desc).*",
            r"^(Crusader|ArcCorp|Hurston|microTech|Orison|Area18|Lorville|NewBabbage)(?!.*_Desc).*",
            r"^(Port_Olisar|GrimHex|Everus|Baijini|Seraphim).*",
            r".*(?:Station|Port|Outpost|Settlement)(?!.*_Desc).*",
        ],
        "vehicle": [
            r"^vehicle_Name.*",
        ],
        "item": [
            r"^item_Name.*",
        ],
        "subtitle": [
            r"^DXSH_",
            r"^Dlg_SC_.*",
            r"^FW22_NT_Datapad_.*",
            r"^FleetWeek2950_.*",
            r"^GenResponse_.*",
            r"^GenericLanding_.*",
            r"^IT_Shared_.*",
            r"^Imperilled_.*",
            r"^MKTG_CUSTOMS1_CV_Access_.*",
            r"^PH_PU_.*",
            r"^PU_.*",
            r"^Pacheco_.*",
            r"^SC_ac_.*",
            r"^SC_lz_.*",
            r"^SM_SIMANN1_.*",
            r"^contract_.*",
            r"^covalex_.*",
            r"^covalexrand_.*",
            r"^covalexspec_.*",
        ],
        "mission": [
            r".*(bounty|Bounty|mission|Mission|contract|Contract).*",
        ],
    }

    @classmethod
    def classify(cls, key: str) -> CategoryResult:
        for category, patterns in cls.PRIORITY_CATEGORIES.items():
            for pattern in patterns:
                if re.match(pattern, key, re.IGNORECASE):
                    return CategoryResult(category=category, is_priority=True)
        return CategoryResult(category="other", is_priority=False)
