import pandas as pd
import re

GROUP_TO_CLASSES = {
    "資訊學群": ["資訊工程", "數據統計", "電機工程", "光電工程", "電子工程", "通訊工程", "生物資訊", "資訊傳播", "圖書資訊", "數位學習", "資訊管理", "電子商務", "媒體設計"],
    "工程學群": ["機械工程", "航空工程", "土木工程", "水利工程", "化學工程", "材料工程", "工程科學", "環境工程", "建築", "運輸物流", "科技管理", "工程不分系", "電資不分系"],
    "數理化學群": ["數學", "化學", "物理", "自然科學", "生化", "財金統計", "理學不分系", "數據統計"],
    "醫藥衛生學群": ["醫學", "公共衛生", "牙醫", "物理治療", "職能治療", "護理", "醫學檢驗", "影像放射", "藥學", "食品營養", "呼吸治療", "健康照護", "化妝品", "職業安全", "視光", "語療聽力", "醫務管理", "獸醫"],
    "生命科學學群": ["生命科學", "生物科技", "生態", "食品生技"],
    "生物資源學群": ["植物保護", "農藝", "動物科學", "園藝", "森林", "海洋資源"],
    "地球環境學群": ["地球科學", "地理", "海洋科學", "大氣科學", "防災", "史地"],
    "建築設計學群": ["都市計畫", "空間設計", "工業設計", "工藝", "商業設計", "服裝設計", "藝術設計", "藝術不分系"],
    "藝術學群": ["美術", "音樂", "表演藝術", "舞蹈"],
    "社會心理學群": ["心理", "社會學", "社會工作", "人類民族", "兒童家庭", "輔導諮商", "勞工關係"],
    "大眾傳播學群": ["大眾傳播", "廣電電影", "新聞", "廣告公關"],
    "外語學群": ["英語文", "歐語文", "日語文", "東方語文", "英語教育", "華語文教育"],
    "文史哲學群": ["中國語文", "歷史", "哲學", "台灣語文", "宗教", "文化產業"],
    "教育學群": ["教育", "特殊教育", "幼兒教育", "成人教育", "社科教育", "科技教育", "數學教育", "英語教育", "華語文教育"],
    "法政學群": ["法律", "財經法律", "政治", "行政管理", "犯罪防治", "土地資產"],
    "管理學群": ["企業管理", "行銷經營", "國際企業", "觀光事業", "運動管理", "餐旅管理", "休閒管理", "商管不分系", "醫務管理", "科技管理"],
    "財經學群": ["會計", "財務金融", "財稅", "保險", "經濟"],
    "遊憩運動學群": ["體育", "運動保健", "觀光事業", "運動管理", "餐旅管理", "休閒管理"],
    "不分系": ["學院不分系", "不分系", "工程不分系", "藝術不分系", "電資不分系", "商管不分系", "理學不分系"],
}

PROGRAM_SPECS = [
    ("外語學院", "英文系", "英", ["英文"]),
    ("外語學院", "日文系", "日", ["日文"]),
    ("外語學院", "西文系", "西", ["西文"]),
    ("人文暨社會科學學院", "中文系", "中", ["中文"]),
    ("人文暨社會科學學院", "社工系", "社工", []),
    ("人文暨社會科學學院", "台文系", "台文", []),
    ("人文暨社會科學學院", "法律系", "法律", []),
    ("人文暨社會科學學院", "大傳系", "大傳", []),
    ("人文暨社會科學學院", "生態系", "生態", []),
    ("人文暨社會科學學院", "法律原住民專班", "法律原專", []),
    ("人文暨社會科學學院", "社工原住民專班", "社工原專", []),
    ("理學院", "財工系", "財工", []),
    ("理學院", "應化系", "應化", []),
    ("理學院", "食營系", "食營", []),
    ("理學院", "化科系", "化科", []),
    ("理學院", "永續環境與智慧科技學士學位學程", "永續", ["永續智慧"]),
    ("管理學院", "行銷與數位經營學系", "行銷", ["行銷與數位經營"]),
    ("管理學院", "國企系", "國企", []),
    ("管理學院", "會計系", "會計", []),
    ("管理學院", "觀光系", "觀光", []),
    ("管理學院", "財金系", "財金", []),
    ("資訊學院", "資管系", "資管", []),
    ("資訊學院", "資工系", "資工", []),
    ("資訊學院", "人工智慧系", "人工智慧", []),
    ("資訊學院", "資科系", "資科", []),
    ("國際學院", "國際資訊學士學位學程", "國際", ["國際資訊"]),
    ("國際學院", "寰宇外語教育學士學位學程", "寰宇外語", ["寰宇外語教育"]),
    ("國際學院", "寰宇管理學士學位學程", "寰宇管理", ["寰宇管理學程"]),
]

COLLEGE_ORDER = list(dict.fromkeys(college for college, _, _, _ in PROGRAM_SPECS))
DEPARTMENT_ORDER = [department for _, department, _, _ in PROGRAM_SPECS]
PREFIX_ORDER = [prefix for _, _, prefix, _ in PROGRAM_SPECS]
PREFIX_TO_COLLEGE = {prefix: college for college, _, prefix, _ in PROGRAM_SPECS}

_MATCH_RULES = []
for college, department, prefix, aliases in PROGRAM_SPECS:
    for alias in [prefix, department, *aliases]:
        _MATCH_RULES.append((alias, college, department, prefix))

for alias, college, department, prefix in [
    ("犯防原專", "人文暨社會科學學院", "犯罪防治原住民專班", "犯防原專"),
    ("犯防", "人文暨社會科學學院", "犯罪防治學系", "犯防"),
    ("經管進", "管理學院", "經營管理進修學士班", "經管進"),
    ("智慧媒體學程", "資訊學院", "智慧媒體學程", "智慧媒體學程"),
    ("晶片設計", "資訊學院", "晶片設計學程", "晶片設計"),
]:
    PREFIX_TO_COLLEGE[alias] = college
    _MATCH_RULES.append((alias, college, department, prefix))

_MATCH_RULES.sort(key=lambda row: len(row[0]), reverse=True)


def _normalize_class_name(class_name: str) -> str:
    s = str(class_name).strip()
    if not s or s.lower() == "nan":
        return ""
    s = re.sub(r"(?:[一二三四五六七八九十]|[1-9])\s*[A-Z]?$", "", s)
    return s.strip()


def get_class_info(class_name: str, unknown: str = "未分類") -> dict:
    base = _normalize_class_name(class_name)
    if not base:
        return {"prefix": unknown, "department": unknown, "college": unknown}
    for alias, college, department, prefix in _MATCH_RULES:
        if base.startswith(alias):
            return {"prefix": prefix, "department": department, "college": college}
    return {"prefix": base, "department": unknown, "college": unknown}


def add_prefix_column(
    df: pd.DataFrame,
    class_col: str = "班級",
    out_col: str = "前綴",
    unknown: str = "未分類",
) -> pd.DataFrame:
    if class_col not in df.columns:
        return df
    df[out_col] = df[class_col].apply(lambda x: get_class_info(x, unknown=unknown)["prefix"])
    return df


def add_department_column(
    df: pd.DataFrame,
    class_col: str = "班級",
    out_col: str = "學系",
    unknown: str = "未分類",
) -> pd.DataFrame:
    if class_col not in df.columns:
        return df
    df[out_col] = df[class_col].apply(lambda x: get_class_info(x, unknown=unknown)["department"])
    return df


def add_college_column(
    df: pd.DataFrame,
    class_col: str = "班級",
    out_col: str = "學院",
    prefix_to_college: dict = None,
    unknown: str = "未分類",
) -> pd.DataFrame:
    if class_col not in df.columns:
        return df
    if prefix_to_college is None:
        prefix_to_college = PREFIX_TO_COLLEGE
    df["班級前綴"] = df[class_col].apply(lambda x: get_class_info(x, unknown=unknown)["prefix"])
    df[out_col] = df[class_col].apply(lambda x: get_class_info(x, unknown=unknown)["college"])
    return df
