import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from normalization import (
    COLLEGE_ORDER,
    DEPARTMENT_ORDER,
    PREFIX_ORDER,
    PREFIX_TO_COLLEGE,
    add_college_column,
    add_department_column,
    add_prefix_column,
)

FILE_PATH = "114學年度 大三學習經驗問卷調查_去識別化.xlsx"
DEFAULT_SHEET = None
PCT_OVERALL = "全體：圖示全體=100%"
PCT_WITHIN_GROUP = "分組百分比：各組各自總和=100%"
HELP_DOC_PATH = Path(__file__).with_name("說明文件.txt")
MULTI_SEP_REGEX = r"[;；,、]"

QuestionType = Literal[
    "meta",
    "single_choice",
    "multi_choice",
    "multi_choice_ranked",
    "likert",
    "short_answer",
    "unknown",
]


@dataclass(frozen=True)
class TagResult:
    qtype: QuestionType
    reason: str
    confidence: float


class SurveyColumnTypeTagger:
    DEFAULT_META_HINTS = [
        "ID", "開始時間", "完成時間", "上次修改時間", "填答時間", "IP", "學號", "姓名", "Email"
    ]

    LIKERT_SETS: List[set] = [
        {"是", "否", "不適用"},
        {"滿意", "普通", "不滿意", "不適用"},
        {"滿意", "普通", "不滿意", "不清楚"},
        {"非常滿意", "滿意", "普通", "不滿意", "非常不滿意"},
    ]

    def __init__(self, extra_meta_cols: Optional[List[str]] = None):
        self.extra_meta_cols = set(extra_meta_cols or [])

    def tag_column(self, df: pd.DataFrame, col: str) -> TagResult:
        header = self._norm(col)

        if col in self.extra_meta_cols:
            return TagResult("meta", "user-provided meta column", 1.0)

        if any(h in header for h in self.DEFAULT_META_HINTS):
            return TagResult("meta", "header looks like metadata", 0.85)

        s = df[col].dropna().astype(str)
        sample = s.head(400)
        uniq = set(sample.unique())
        uniq_norm = {self._norm(v) for v in uniq}

        n_total = len(s)
        n_resp = len(sample)
        resp_rate = n_resp / max(n_total, 1)
        u = len(uniq_norm)

        if resp_rate <= 0.5 and 2 <= u <= max(20, int(0.5 * n_resp)):
            return TagResult(
                "single_choice",
                f"sparse responses: resp_rate={resp_rate:.2f}, uniq={u}",
                0.85,
            )

        if self._has_any(header, ["需排序", "排序", "rank", "順位", "最主要", "第二次要", "第三次要"]):
            if self._looks_delimited_multi(sample):
                return TagResult("multi_choice_ranked", "header indicates ranking + delimited values", 0.95)
            return TagResult("multi_choice_ranked", "header indicates ranking", 0.8)

        if self._has_any(header, ["可複選", "(可複選)", "multiple choice"]):
            if self._looks_delimited_multi(sample):
                return TagResult("multi_choice", "header says multi + delimited values", 0.97)
            return TagResult("multi_choice", "header says multi-choice", 0.9)

        for lk in self.LIKERT_SETS:
            if uniq_norm.issubset({self._norm(x) for x in lk}) and len(uniq_norm) >= 2:
                return TagResult("likert", "values match a likert option set", 0.98)

        if self._looks_delimited_multi(sample):
            return TagResult("multi_choice", "values look like multi selections delimited", 0.8)

        if self._has_any(header, ["請提供", "意見", "建議", "最喜歡", "請列出", "課程名稱", "please provide", "please list", "feedback", "suggestions"]):
            return TagResult("short_answer", "header looks open-ended", 0.8)

        if n_resp > 0:
            uniq_ratio = u / n_resp
            avg_len = sample.astype(str).str.len().mean() if len(sample) else 0
            max_u = min(30, max(12, int(0.2 * n_resp)))
            if 2 <= u <= max_u and uniq_ratio <= 0.6:
                return TagResult(
                    "single_choice",
                    f"categorical pattern: uniq={u}, uniq_ratio={uniq_ratio:.2f}",
                    0.78,
                )

        if self._looks_free_text(sample):
            return TagResult("short_answer", "values look like free text", 0.65)

        return TagResult("unknown", "no strong signal", 0.3)

    @staticmethod
    def _norm(x: str) -> str:
        x = str(x).strip()
        x = re.sub(r"\s+", " ", x)
        return x

    @staticmethod
    def _has_any(text: str, needles: List[str]) -> bool:
        return any(n in text for n in needles)

    @staticmethod
    def _looks_delimited_multi(values: pd.Series) -> bool:
        if len(values) == 0:
            return False
        v = values.astype(str)
        hit = v.str.contains(MULTI_SEP_REGEX, regex=True).mean()
        return hit >= 0.15

    @staticmethod
    def _looks_free_text(values: pd.Series) -> bool:
        if len(values) == 0:
            return False
        v = values.astype(str)
        avg_len = v.str.len().mean()
        uniq_ratio = v.nunique() / max(len(v), 1)
        return (avg_len >= 12 and uniq_ratio >= 0.5)


@st.cache_data
def load_excel(path: str, sheet: Optional[str]):
    if sheet:
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_excel(path)
    df = df.dropna(axis=1, how="all")
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda c: c.astype(str).str.strip().replace({"nan": np.nan}))
    return df


def explode_multi(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(str)
    parts = s.str.split(MULTI_SEP_REGEX, regex=True)
    out = parts.explode().str.strip()
    return out[out.notna() & (out != "")]


def ranked_stats(
    series: pd.Series,
    group: Optional[pd.Series] = None,
    top_k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s = series.dropna().astype(str)
    rank_rows = []
    rank_sum = {}
    mention_count = {}

    if group is not None:
        group = group.loc[s.index].astype(str).fillna("(未分類)")

    for idx, cell in s.items():
        items = [x.strip() for x in re.split(MULTI_SEP_REGEX, cell) if x.strip()]
        items = items[:top_k]
        g = group.loc[idx] if group is not None else None
        for r, opt in enumerate(items, start=1):
            if g is not None:
                rank_rows.append((g, opt, r))
                key = (g, opt)
            else:
                rank_rows.append((opt, r))
                key = opt
            rank_sum[key] = rank_sum.get(key, 0) + r
            mention_count[key] = mention_count.get(key, 0) + 1

    if group is not None:
        rank_df = pd.DataFrame(rank_rows, columns=["group", "option", "rank"])
        if rank_df.empty:
            rank_table = pd.DataFrame()
        else:
            rank_table = (
                rank_df.groupby(["group", "option", "rank"]).size().unstack(fill_value=0).reset_index()
            )
        score_df = pd.DataFrame(
            {
                "group": [g for g, _ in rank_sum.keys()],
                "option": [o for _, o in rank_sum.keys()],
                "總順位": [rank_sum[k] for k in rank_sum],
                "平均順位": [rank_sum[k] / mention_count[k] for k in rank_sum],
                "被提及次數": [mention_count[k] for k in rank_sum],
            }
        )
        score_df = score_df.sort_values(by=["group", "總順位", "平均順位"], ascending=True).reset_index(drop=True)
    else:
        rank_df = pd.DataFrame(rank_rows, columns=["option", "rank"])
        if rank_df.empty:
            rank_table = pd.DataFrame()
        else:
            rank_table = (
                rank_df.groupby(["option", "rank"]).size().unstack(fill_value=0).reset_index()
            )
        score_df = pd.DataFrame(
            {
                "option": list(rank_sum.keys()),
                "總順位": [rank_sum[o] for o in rank_sum],
                "平均順位": [rank_sum[o] / mention_count[o] for o in rank_sum],
                "被提及次數": [mention_count[o] for o in rank_sum],
            }
        )
        score_df = score_df.sort_values(by=["總順位", "平均順位"], ascending=True).reset_index(drop=True)

    return rank_table, score_df


def group_value_counts(df: pd.DataFrame, col: str, group_col: Optional[str] = None) -> pd.DataFrame:
    if group_col and group_col in df.columns:
        tmp = df[[group_col, col]].dropna()
        out = tmp.groupby([group_col, col]).size().reset_index(name="count")
        return out
    out = df[col].dropna().value_counts().reset_index()
    out.columns = ["value", "count"]
    return out


def add_percent(df_counts: pd.DataFrame, count_col: str = "count", group_col: Optional[str] = None, overall: bool = False) -> pd.DataFrame:
    out = df_counts.copy()
    if group_col and group_col in out.columns:
        if overall:
            total = out[count_col].sum()
            out["percent"] = out[count_col] / max(total, 1) * 100
        else:
            denom = out.groupby(group_col)[count_col].transform("sum")
            out["percent"] = out[count_col] / denom * 100
    else:
        out["percent"] = out[count_col] / max(out[count_col].sum(), 1) * 100
    return out


def parse_class_key(class_name: str, college_order=None):
    text = str(class_name or "").strip()
    if not text:
        return (len(college_order) if college_order is not None else 999, len(PREFIX_ORDER), "", 0, "")

    college = ""
    prefix = ""
    try:
        parts = text.split()
        prefix = parts[0]
    except Exception:
        prefix = text

    college_rank = college_order.index(college) if college_order and college in college_order else (len(college_order) if college_order else 999)
    prefix_rank = PREFIX_ORDER.index(prefix) if prefix in PREFIX_ORDER else len(PREFIX_ORDER)

    m = re.search(r'([一二三四1234])(?:年級)?\s*([A-Za-z])(?:班)?$', text)
    if m:
        year_str = m.group(1)
        class_str = m.group(2)
        year_num = {'一': 1, '二': 2, '三': 3, '四': 4, '1': 1, '2': 2, '3': 3, '4': 4}.get(year_str, 0)
        return (college_rank, prefix_rank, prefix, year_num, class_str)
    return (college_rank, prefix_rank, prefix, 0, text)


def try_numeric_order(values) -> Optional[List[str]]:
    vals = [str(v) for v in values if str(v) not in ("", "nan")]
    if len(vals) < 2:
        return None
    def _leading_num(text: str) -> float:
        text = str(text).strip()
        if re.search(r"沒有|無工讀|no part", text, re.IGNORECASE):
            return -1.0
        m = re.search(r"\d+", text)
        return float(m.group()) if m else float("inf")
    keys = [_leading_num(v) for v in vals]
    has_num = sum(1 for k in keys if k not in (-1.0, float("inf")))
    if has_num / len(vals) >= 0.4:
        return [v for _, v in sorted(zip(keys, vals))]
    return None


def apply_normalized_order(result: pd.DataFrame, col: str, college_order, class_order=None):
    if col not in result.columns:
        return result
    if col == "學院":
        order = college_order
    elif col == "班級":
        if class_order is None:
            unique_classes = result[col].dropna().astype(str).unique()
            class_order = sorted(unique_classes, key=lambda c: parse_class_key(c, college_order))
        order = class_order
    elif col == "前綴":
        order = [x for x in PREFIX_ORDER if x in result[col].dropna().astype(str).unique()]
    elif col == "學系":
        order = [x for x in DEPARTMENT_ORDER if x in result[col].dropna().astype(str).unique()]
    else:
        return result
    result[col] = pd.Categorical(result[col].astype(str), categories=order, ordered=True)
    return result.sort_values([col] + [c for c in result.columns if c != col])


def get_percent_column_label(pct_mode: Optional[str], group_label: str) -> str:
    if pct_mode == PCT_OVERALL:
        return "百分比（全體=100%）"
    if group_label != "(不分組)":
        return "百分比（各組=100%）"
    return "百分比"


def normalize_display_table(df: pd.DataFrame, percent_col_label: str = "百分比") -> pd.DataFrame:
    out = df.copy()
    if "option" in out.columns:
        out = out.rename(columns={"option": "選項"})
    if "value" in out.columns:
        out = out.rename(columns={"value": "選項"})
    if "count" in out.columns:
        out = out.rename(columns={"count": "人數"})
    if "percent" in out.columns:
        out = out.rename(columns={"percent": percent_col_label})
        out[percent_col_label] = out[percent_col_label].astype(float).round(2).map(lambda x: f"{x:.2f}%")
    return out


def show_table(df: pd.DataFrame, percent_col_label: str = "百分比", **kwargs):
    st.dataframe(normalize_display_table(df, percent_col_label=percent_col_label), hide_index=True, width="stretch", **kwargs)


def build_population_text(selected_colleges: List[str], selected_departments: List[str], selected_classes: List[str]) -> str:
    parts: List[str] = []
    if selected_colleges:
        parts.append("、".join(selected_colleges))
    if selected_departments:
        parts.append("、".join(selected_departments))
    if selected_classes:
        parts.append("、".join(selected_classes))
    return "；".join(parts)


def build_table_caption(
    question_label: str,
    group_label: str,
    selected_colleges: List[str],
    selected_departments: List[str],
    selected_classes: List[str],
) -> str:
    population_text = build_population_text(selected_colleges, selected_departments, selected_classes)
    grouped = group_label != "(不分組)"
    filtered = bool(selected_colleges or selected_departments or selected_classes)
    if grouped and filtered:
        return f"{question_label}依{group_label}篩選在{population_text}統計"
    if grouped:
        return f"{question_label}依{group_label}的全校統計"
    if filtered:
        return f"{question_label}在{population_text}的統計"
    return f"{question_label}的全校統計"


def load_help_document(path: Path = HELP_DOC_PATH) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "尚未建立說明文件。"


def summarize_question(
    df: pd.DataFrame,
    question: str,
    qtype: str,
    group_label: str,
    pct_mode: Optional[str],
) -> pd.DataFrame:
    if qtype == "multi_choice":
        exploded = explode_multi(df[question])
        d = exploded.to_frame(name=question)
        d[group_label] = df.loc[exploded.index, group_label] if group_label in df.columns else np.nan
    else:
        d = df[[question]].copy()
        d = d[d[question].notna() & (d[question].astype(str).str.strip() != "")]
        if group_label in df.columns:
            d[group_label] = df.loc[d.index, group_label]

    if group_label != "(不分組)" and group_label in d.columns:
        out = d.groupby([group_label, question]).size().reset_index(name="count")
        if pct_mode is not None:
            out = add_percent(out, count_col="count", group_col=group_label, overall=(pct_mode == PCT_OVERALL))
    else:
        out = d[question].astype(str).value_counts(dropna=True).reset_index()
        out.columns = [question, "count"]
        if pct_mode is not None:
            out = add_percent(out, count_col="count", group_col=None)
    return out


def build_chart(df: pd.DataFrame, question: str, group_label: str, qtype: str, pct_mode: Optional[str]) -> None:
    if qtype == "short_answer":
        return

    y_col = "percent" if pct_mode is not None else "count"
    text_template = "%{text:.2f}%" if y_col == "percent" else "%{text:.0f}"
    y_axis_label = "百分比(%)" if y_col == "percent" else "人數"

    legend_kwargs = dict(itemwidth=220, itemsizing="constant")
    if group_label != "(不分組)" and group_label in df.columns:
        fig = px.bar(
            df,
            x=question,
            y=y_col,
            color=group_label,
            barmode="group",
            text=y_col,
        )
        fig.update_layout(legend_title=group_label, legend=legend_kwargs)
    else:
        fig = px.bar(
            df,
            x=question,
            y=y_col,
            text=y_col,
        )
        fig.update_layout(legend=legend_kwargs)

    fig.update_layout(xaxis_title=question, yaxis_title=y_axis_label)
    fig.update_traces(texttemplate=text_template, textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def build_rank_trend_chart(rank_df: pd.DataFrame, group_label: str) -> None:
    if rank_df.empty or "group" not in rank_df.columns:
        return
    fig = px.bar(
        rank_df,
        x="group",
        y="平均順位",
        color="option",
        barmode="group",
        title="分組排名趨勢圖：平均順位",
        text="平均順位",
    )
    fig.update_layout(
        xaxis_title=group_label,
        yaxis_title="平均順位（數字越小越重要）",
        legend_title="選項",
        legend=dict(itemwidth=220, itemsizing="constant"),
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="115學年度大三學習經驗調查分析", layout="wide")

st.title("114學年度大三學習經驗問卷調查 - 去識別化分析")

if "show_help_doc" not in st.session_state:
    st.session_state.show_help_doc = False

if st.session_state.show_help_doc:
    st.subheader("說明文件")
    st.markdown(load_help_document())
    st.divider()


df = load_excel(FILE_PATH, DEFAULT_SHEET)

if "班級" in df.columns:
    df = add_prefix_column(df, class_col="班級", out_col="前綴", unknown="未分類")
    df = add_department_column(df, class_col="班級", out_col="學系", unknown="未分類")
    df = add_college_column(
        df,
        class_col="班級",
        out_col="學院",
        prefix_to_college=PREFIX_TO_COLLEGE,
        unknown="未分類",
    )

if df.empty:
    st.error("無法讀取資料，請確認 Excel 檔案是否存在且不是空表。")
    st.stop()

selected_colleges: List[str] = []
selected_departments: List[str] = []
selected_classes: List[str] = []
pct_mode: Optional[str] = None
population_attrs: List[str] = []

# infer types
tagger = SurveyColumnTypeTagger()
type_rows = []
for c in df.columns:
    res = tagger.tag_column(df, c)
    type_rows.append({"column": c, "qtype": res.qtype, "confidence": res.confidence, "reason": res.reason})
types_df = pd.DataFrame(type_rows)

meta_candidates = types_df.loc[types_df["qtype"] == "meta", "column"].tolist()
group_candidates = meta_candidates.copy()
if "學院" in df.columns and "學院" not in group_candidates:
    group_candidates.insert(0, "學院")
extra_group_candidates = types_df.loc[types_df["qtype"].isin(["single_choice", "likert"]), "column"].tolist()
for x in extra_group_candidates:
    if x not in group_candidates:
        group_candidates.append(x)
excluded_group_fields = {"ID", "開始時間", "完成時間"}
group_candidates = [c for c in group_candidates if c not in excluded_group_fields]

excluded_question_fields = {"前綴", "學系", "學院"}
question_cols = types_df.loc[
    ~types_df["qtype"].isin(["meta"]) & ~types_df["column"].isin(excluded_question_fields),
    "column",
].tolist()

with st.sidebar:
    st.header("分析設定")
    question = st.selectbox("問卷題目（圖表類別）", question_cols)
    available_group_options = [opt for opt in ["(不分組)"] + group_candidates if opt == "(不分組)" or opt != question]
    group_label = st.selectbox("分組比較（群組標籤）", available_group_options, index=0)

    st.divider()

    population_attrs = st.multiselect(
        "學院、系、班級篩選（可複選交叉比對或留空表示全校）",
        ["學院", "學系", "班級"],
        default=[],
        placeholder="不篩選(全校)",
    )

    if "學院" in population_attrs and "學院" in df.columns:
        college_values = [x for x in COLLEGE_ORDER if x in df["學院"].dropna().astype(str).unique()]
        extras = [x for x in sorted(df["學院"].dropna().astype(str).unique()) if x not in college_values]
        selected_colleges = st.multiselect(
            "選取學院（可多選）",
            college_values + extras,
            default=[],
            placeholder="不篩選(全校)",
        )

    if "學系" in population_attrs and "學系" in df.columns:
        department_values = [x for x in DEPARTMENT_ORDER if x in df["學系"].dropna().astype(str).unique()]
        department_extras = [x for x in sorted(df["學系"].dropna().astype(str).unique()) if x not in department_values]
        selected_departments = st.multiselect(
            "選取學系（可多選）",
            department_values + department_extras,
            default=[],
            placeholder="不篩選(全校)",
        )

    if "班級" in population_attrs and "班級" in df.columns:
        class_values = df["班級"].dropna().astype(str).unique()
        class_ordered = sorted(class_values, key=lambda c: parse_class_key(c, COLLEGE_ORDER))
        selected_classes = st.multiselect(
            "選取班級（可多選）",
            class_ordered,
            default=[],
            placeholder="不篩選(全校)",
        )

    st.divider()

    show_pct = st.checkbox("顯示百分比 (%)", value=False)
    if show_pct:
        pct_mode = st.radio(
            "百分比母體",
            [PCT_OVERALL, PCT_WITHIN_GROUP],
            index=1,
        )

    st.divider()
    if st.button("說明文件", use_container_width=True):
        st.session_state.show_help_doc = not st.session_state.show_help_doc
    st.caption(f"文件：{HELP_DOC_PATH.name}")

    group_col = None if group_label == "(不分組)" else group_label

mask = pd.Series(True, index=df.index)
if population_attrs:
    if selected_colleges or selected_departments or selected_classes:
        mask = pd.Series(False, index=df.index)
        if selected_colleges:
            mask |= df["學院"].isin(selected_colleges)
        if selected_departments:
            mask |= df["學系"].isin(selected_departments)
        if selected_classes:
            mask |= df["班級"].isin(selected_classes)
    else:
        mask = pd.Series(False, index=df.index)

if not mask.any():
    st.warning("篩選後無資料，請調整學院/班級篩選條件。")

df = df[mask]

if group_col is None:
    group_label = "(不分組)"

if df.empty:
    st.warning("篩選後無資料，請調整學院/班級篩選條件。")
    st.stop()

question_type = types_df.loc[types_df["column"] == question, "qtype"].iloc[0]

st.markdown(f"**資料筆數**：{len(df)}，**題目類型**：{question_type}")

if question_type == "multi_choice_ranked":
    st.subheader(f"{question}：排序結果")
    group_series = None
    if group_label != "(不分組)" and group_label in df.columns:
        group_series = df[group_label]
    rank_table, score_table = ranked_stats(df[question], group=group_series)
    if not rank_table.empty:
        st.markdown("**排名次數**")
        show_table(rank_table)
    if not score_table.empty:
        st.markdown("**順位總和與平均順位（數字越小表示越重要）**")
        show_table(score_table)
    if group_series is not None and not score_table.empty:
        st.markdown("**分組排名趨勢圖**")
        build_rank_trend_chart(score_table, group_label)
else:
    result = summarize_question(df, question, question_type, group_label, pct_mode)
    caption = build_table_caption(question, group_label, selected_colleges, selected_departments, selected_classes)
    st.subheader(caption)
    percent_label = get_percent_column_label(pct_mode if pct_mode != "不顯示百分比" else None, group_label)
    show_table(result, percent_col_label=percent_label)

    if question_type != "short_answer":
        build_chart(result, question, group_label, question_type, pct_mode)
    else:
        st.markdown("---")
        st.markdown("### 簡答題樣本（最多 20 筆）")
        sample_values = df[question].dropna().astype(str).str.strip()
        sample_values = sample_values[sample_values != ""]
        if sample_values.empty:
            st.write("目前無有效回答資料。")
        else:
            for value in sample_values.sample(min(20, len(sample_values)), random_state=1):
                st.write(f"- {value}")
