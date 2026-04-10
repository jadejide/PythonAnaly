from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from integrated_irt_mirt_pipeline import (
    MIRTPrepConfig,
    build_student_knowledge_report,
    create_work_dir,
    generate_pipeline,
)
from irt_pipeline_pandas import (
    IRTConfig,
    icc_curve,
    iif_curve,
    test_information_curve,
    test_se_curve,
)


st.set_page_config(
    page_title="IRT + MIRT 学情分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_white"
ACCENT = "#4F46E5"
ACCENT_2 = "#06B6D4"
ACCENT_3 = "#F59E0B"
ACCENT_4 = "#10B981"
BG_SOFT = "#F8FAFC"
CARD_BG = "rgba(255,255,255,0.82)"
BORDER = "rgba(99,102,241,0.10)"


# =========================
# 基础工具
# =========================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(79,70,229,0.10), transparent 26%),
                radial-gradient(circle at top right, rgba(6,182,212,0.08), transparent 22%),
                linear-gradient(180deg, #f8fbff 0%, #f6f8fc 100%);
        }}
        .block-container {{padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px;}}
        .hero {{
            padding: 1.35rem 1.4rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(79,70,229,0.94) 0%, rgba(6,182,212,0.88) 100%);
            color: white;
            box-shadow: 0 18px 50px rgba(79,70,229,0.16);
            margin-bottom: 0.9rem;
        }}
        .hero h1 {{font-size: 2rem; margin: 0 0 0.35rem 0;}}
        .hero p {{font-size: 0.98rem; opacity: 0.95; margin: 0;}}
        .pill-row {{display: flex; gap: 8px; flex-wrap: wrap; margin-top: 0.75rem;}}
        .pill {{
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.22);
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
        }}
        .note-card {{
            background: {CARD_BG};
            backdrop-filter: blur(10px);
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 26px rgba(15,23,42,0.05);
            margin-bottom: 0.8rem;
        }}
        .mini-card {{
            background: {CARD_BG};
            backdrop-filter: blur(8px);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            box-shadow: 0 10px 24px rgba(15,23,42,0.04);
        }}
        .subtle {{color: #475569; font-size: 0.92rem;}}
        .section-title {{font-size: 1.1rem; font-weight: 700; margin: 0.1rem 0 0.6rem 0;}}
        div[data-testid="stMetric"] {{
            background: {CARD_BG};
            border: 1px solid rgba(148,163,184,0.18);
            padding: 0.7rem 0.85rem;
            border-radius: 18px;
            box-shadow: 0 10px 22px rgba(15,23,42,0.04);
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.16);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #fbfdff 0%, #f7f9fd 100%);
            border-right: 1px solid rgba(148,163,184,0.16);
        }}
        .footer-note {{color:#64748B; font-size:0.86rem; margin-top:0.6rem;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.88)",
        margin=dict(l=20, r=20, t=52, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def pretty_pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{float(x):.2%}"


def df_download_button(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label,
        df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        filename,
        "text/csv",
        use_container_width=True,
    )


def json_download_button(data: Any, label: str, filename: str) -> None:
    st.download_button(
        label,
        json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        filename,
        "application/json",
        use_container_width=True,
    )


def get_score_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("F") and not c.startswith("SE_")]


def make_closed_radar(categories: List[str], values: List[float], name: str, max_range: float = 100.0) -> go.Figure:
    fig = go.Figure()
    if categories:
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=name,
                line=dict(color=ACCENT, width=3),
                fillcolor="rgba(79,70,229,0.22)",
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max_range], gridcolor="rgba(148,163,184,0.25)")),
        showlegend=False,
    )
    return style_fig(fig, 420)


# =========================
# 数据处理
# =========================
def build_test_curve_df(item_param_df: pd.DataFrame) -> pd.DataFrame:
    if item_param_df.empty:
        return pd.DataFrame()
    usable = item_param_df.copy()
    if "qid" not in usable.columns and "original_item_id" in usable.columns:
        usable = usable.rename(columns={"original_item_id": "qid"})
    usable = usable[[c for c in ["qid", "a", "b"] if c in usable.columns]].dropna()
    if usable.empty:
        return pd.DataFrame()
    theta_grid = np.linspace(-4, 4, 161)
    return pd.DataFrame(
        {
            "theta": theta_grid,
            "TIF": test_information_curve(theta_grid, usable),
            "SE": test_se_curve(theta_grid, usable),
        }
    )


def build_item_curve_df(item_param_df: pd.DataFrame, qid: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    tmp = item_param_df.copy()
    if "qid" not in tmp.columns and "original_item_id" in tmp.columns:
        tmp = tmp.rename(columns={"original_item_id": "qid"})
    if "qid" not in tmp.columns:
        return pd.DataFrame(), None
    row_df = tmp[tmp["qid"].astype(str) == str(qid)]
    if row_df.empty:
        return pd.DataFrame(), None
    row = row_df.iloc[0]
    a = safe_float(row.get("a"), 1.0)
    b = safe_float(row.get("b"), 0.0)
    theta_grid = np.linspace(-4, 4, 161)
    return pd.DataFrame(
        {"theta": theta_grid, "ICC": icc_curve(theta_grid, a, b), "IIF": iif_curve(theta_grid, a, b)}
    ), row


def compute_pca_2d(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = get_score_columns(df)
    if not score_cols:
        return pd.DataFrame()
    X = df[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if X.shape[0] < 2 or X.shape[1] < 1:
        return pd.DataFrame()
    X = X - X.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    out = pd.DataFrame({"student_id": df["student_id"].astype(str)})
    out["PC1"] = u[:, 0] * s[0]
    out["PC2"] = u[:, 1] * s[1] if X.shape[1] > 1 else 0.0
    out["ability_mean"] = df[score_cols].mean(axis=1)
    return out


def simple_kmeans_labels(df: pd.DataFrame, k: int = 3, n_iter: int = 20) -> pd.Series:
    score_cols = get_score_columns(df)
    if len(score_cols) < 1 or len(df) < k:
        return pd.Series(["群组1"] * len(df), index=df.index)
    X = df[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    X = (X - X.mean(axis=0, keepdims=True)) / np.where(X.std(axis=0, keepdims=True) == 0, 1, X.std(axis=0, keepdims=True))
    centers = X[np.linspace(0, len(X) - 1, k, dtype=int)].copy()
    for _ in range(n_iter):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for idx in range(k):
            grp = X[labels == idx]
            if len(grp) > 0:
                centers[idx] = grp.mean(axis=0)
    return pd.Series([f"群组{int(x)+1}" for x in labels], index=df.index)


def build_cohort_factor_df(factor_scores_df: pd.DataFrame, factor_meta_df: pd.DataFrame) -> pd.DataFrame:
    if factor_scores_df.empty or factor_meta_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, meta in factor_meta_df.iterrows():
        fcol = str(meta["factor_name"])
        if fcol not in factor_scores_df.columns:
            continue
        s = pd.to_numeric(factor_scores_df[fcol], errors="coerce")
        rows.append(
            {
                "factor_name": fcol,
                "mid_name": str(meta.get("mid_name", fcol)),
                "mid_id": str(meta.get("mid_id", "")),
                "mean_score": float(s.mean()),
                "std_score": float(s.std(ddof=0)),
                "median_score": float(s.median()),
                "n_items": int(meta.get("n_items", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_score")


def assign_performance_band(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return series.apply(lambda x: "重点关注" if x <= q1 else ("优势稳定" if x >= q3 else "中间区间"))


def build_factor_corr_df(factor_scores_df: pd.DataFrame) -> pd.DataFrame:
    cols = get_score_columns(factor_scores_df)
    if len(cols) < 2:
        return pd.DataFrame()
    return factor_scores_df[cols].corr(numeric_only=True)


def rename_factor_labels(corr: pd.DataFrame, factor_meta_df: pd.DataFrame) -> pd.DataFrame:
    if corr.empty:
        return corr
    name_map = {}
    if not factor_meta_df.empty:
        name_map = dict(zip(factor_meta_df["factor_name"].astype(str), factor_meta_df["mid_name"].astype(str)))
    out = corr.copy()
    out.index = [name_map.get(str(x), str(x)) for x in out.index]
    out.columns = [name_map.get(str(x), str(x)) for x in out.columns]
    return out


def get_item_merged_df(run: Dict[str, Any]) -> pd.DataFrame:
    item_param_df = run["irt"]["item_param_df"].copy()
    item_stats_df = run["irt"]["item_stats_df"].copy()
    if item_param_df.empty:
        return item_param_df
    if "qid" not in item_param_df.columns and "original_item_id" in item_param_df.columns:
        item_param_df = item_param_df.rename(columns={"original_item_id": "qid"})
    if not item_stats_df.empty and "qid" in item_stats_df.columns:
        item_param_df = item_param_df.merge(item_stats_df, on="qid", how="left")
    return item_param_df


def build_student_overview_cards(run: Dict[str, Any]) -> Dict[str, Any]:
    irt_metrics = run["irt"]["metrics"]
    mirt_summary = run["mirt_prepare"]["summary"]
    item_df = run["irt"]["item_param_df"]
    theta_df = run["irt"]["theta_df"]
    return {
        "学生总数": irt_metrics.get("students_total", 0),
        "题目总数": irt_metrics.get("items_total", 0),
        "作答记录": irt_metrics.get("responses_total", 0),
        "IRT题目": int(len(item_df)) if isinstance(item_df, pd.DataFrame) else 0,
        "MIRT因子": mirt_summary.get("factors", 0),
        "θ均值": round(float(theta_df["theta"].mean()), 3) if isinstance(theta_df, pd.DataFrame) and not theta_df.empty else "-",
    }


# =========================
# 渲染块
# =========================
def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>📊 IRT + MIRT 学情分析平台</h1>
            <p>自动运行 IRT 与 Python 近似 MIRT，输出学生画像、题目诊断、群体结构和可导出报告。</p>
            <div class="pill-row">
                <span class="pill">学生学情画像</span>
                <span class="pill">题目参数诊断</span>
                <span class="pill">知识点掌握度</span>
                <span class="pill">群体分层聚类</span>
                <span class="pill">一键导出 CSV / JSON</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # st.markdown(
    #     '<div class="note-card"><div class="section-title">这版做了什么</div><div class="subtle">我把你的前端升级成更适合发布到 GitHub 的展示版：首屏更完整、图表更多、层次更清楚，同时保持你现有的 IRT + Python 近似 MIRT 管线不变。</div></div>',
    #     unsafe_allow_html=True,
    # )


def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.header("上传与参数")
        question_bank_file = st.file_uploader("1) 题库 data.json", type=["json"])
        students_file = st.file_uploader("2) 学生聚合 data_aggregation.json", type=["json"])
        taxonomy_file = st.file_uploader("3) 新课标知识点 xlsx", type=["xlsx"])

        st.markdown("### IRT 参数")
        min_item_responses = st.number_input("IRT 最少题目样本", min_value=1, value=30)
        min_student_responses = st.number_input("IRT 最少学生作答数", min_value=1, value=10)
        max_iters = st.number_input("IRT 最大迭代次数", min_value=10, value=120)

        st.markdown("### MIRT 预处理参数")
        min_resp_per_item = st.number_input("MIRT 每题最少作答数", min_value=1, value=30)
        min_resp_per_student = st.number_input("MIRT 每生最少作答数", min_value=1, value=10)
        min_items_per_factor = st.number_input("每个二级知识点最少题量", min_value=1, value=10)

        st.markdown("### 展示选项")
        cluster_k = st.slider("群体聚类数", min_value=2, max_value=5, value=3)
        top_n = st.slider("排行榜展示条数", min_value=5, max_value=30, value=12)
        work_base_dir = st.text_input("本地运行目录（可留空）", value="")
        run_btn = st.button("开始一键运行", type="primary", use_container_width=True)

    return {
        "question_bank_file": question_bank_file,
        "students_file": students_file,
        "taxonomy_file": taxonomy_file,
        "cluster_k": int(cluster_k),
        "top_n": int(top_n),
        "irt_cfg": IRTConfig(
            min_item_responses=int(min_item_responses),
            min_student_responses=int(min_student_responses),
            max_iters=int(max_iters),
        ),
        "mirt_cfg": MIRTPrepConfig(
            min_resp_per_item=int(min_resp_per_item),
            min_resp_per_student=int(min_resp_per_student),
            min_items_per_factor=int(min_items_per_factor),
        ),
        "work_base_dir": work_base_dir,
        "run_btn": run_btn,
    }


def run_pipeline_ui(inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not inputs["run_btn"]:
        return st.session_state.get("latest_run")
    if not (inputs["question_bank_file"] and inputs["students_file"] and inputs["taxonomy_file"]):
        st.error("请先上传完整的 3 个输入文件。")
        return st.session_state.get("latest_run")

    work_dir = create_work_dir(inputs["work_base_dir"] or None)
    progress = st.progress(0)
    status_box = st.empty()
    console_box = st.empty()
    run_result = None

    gen = generate_pipeline(
        question_bank_source=inputs["question_bank_file"],
        students_source=inputs["students_file"],
        taxonomy_source=inputs["taxonomy_file"],
        work_dir=work_dir,
        irt_config=inputs["irt_cfg"],
        mirt_prep_config=inputs["mirt_cfg"],
    )
    try:
        while True:
            update = next(gen)
            progress.progress(float(update.get("progress", 0.0)))
            status_box.info(update.get("message", "运行中..."))
            if update.get("console"):
                console_box.code(update["console"], language="text")
    except StopIteration as e:
        run_result = e.value
    except Exception as e:
        status_box.error(str(e))
        return st.session_state.get("latest_run")

    progress.progress(1.0)
    status_box.success("拟合完成，下面可以查看完整分析结果。")
    st.session_state.latest_run = run_result
    return run_result


def render_summary_cards(run: Dict[str, Any]) -> None:
    cards = build_student_overview_cards(run)
    cols = st.columns(len(cards))
    for col, (name, value) in zip(cols, cards.items()):
        col.metric(name, value)


def render_overview_tab(run: Dict[str, Any], cluster_k: int, top_n: int) -> None:
    irt = run["irt"]
    mirt = run["mirt"]
    theta_df = irt["theta_df"].copy()
    item_df = get_item_merged_df(run)
    factor_scores_df = mirt["student_factor_scores_df"].copy()
    factor_meta_df = mirt["factor_meta_df"].copy()
    cohort_df = build_cohort_factor_df(factor_scores_df, factor_meta_df)

    st.subheader("总览仪表盘")
    a1, a2, a3 = st.columns([1.1, 1.0, 0.9])
    with a1:
        if not theta_df.empty and "theta" in theta_df.columns:
            theta_df["分层"] = assign_performance_band(theta_df["theta"])
            fig = px.histogram(theta_df, x="theta", color="分层", nbins=28, title="学生能力 θ 分布", color_discrete_sequence=[ACCENT, ACCENT_2, ACCENT_3])
            st.plotly_chart(style_fig(fig, 350), use_container_width=True)
        else:
            st.info("暂无 θ 分布。")
    with a2:
        if not item_df.empty and "b" in item_df.columns:
            fig = px.histogram(item_df, x="b", nbins=26, title="题目难度 b 分布", color_discrete_sequence=[ACCENT_2])
            st.plotly_chart(style_fig(fig, 350), use_container_width=True)
        else:
            st.info("暂无题目难度分布。")
    with a3:
        if not item_df.empty and "a" in item_df.columns:
            fig = px.histogram(item_df, x="a", nbins=26, title="题目区分度 a 分布", color_discrete_sequence=[ACCENT_3])
            st.plotly_chart(style_fig(fig, 350), use_container_width=True)
        else:
            st.info("暂无题目区分度分布。")

    b1, b2 = st.columns([1.25, 0.75])
    with b1:
        curve_df = build_test_curve_df(item_df)
        if not curve_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve_df["theta"], y=curve_df["TIF"], mode="lines", name="TIF", line=dict(color=ACCENT, width=3)))
            fig.add_trace(go.Scatter(x=curve_df["theta"], y=curve_df["SE"], mode="lines", name="SE", yaxis="y2", line=dict(color=ACCENT_3, width=3)))
            fig.update_layout(
                title="测验信息函数 TIF 与标准误 SE",
                yaxis=dict(title="TIF"),
                yaxis2=dict(title="SE", overlaying="y", side="right"),
            )
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        else:
            st.info("暂无测验信息函数。")
    with b2:
        if not cohort_df.empty:
            low = cohort_df.iloc[0]
            high = cohort_df.iloc[-1]
            st.markdown(
                f'''<div class="note-card"><div class="section-title">知识点快照</div>
                <div class="subtle">当前群体最弱维度：<b>{low['mid_name']}</b><br>平均能力：<b>{low['mean_score']:.3f}</b></div>
                <hr style="border:none;border-top:1px solid rgba(148,163,184,0.18);margin:0.8rem 0;">
                <div class="subtle">当前群体最强维度：<b>{high['mid_name']}</b><br>平均能力：<b>{high['mean_score']:.3f}</b></div>
                </div>''',
                unsafe_allow_html=True,
            )
        else:
            st.info("暂无群体知识点快照。")

    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        if not cohort_df.empty:
            fig = px.bar(
                cohort_df.sort_values("mean_score"),
                x="mean_score",
                y="mid_name",
                color="n_items",
                orientation="h",
                title="群体知识点平均能力排序",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        else:
            st.info("暂无群体知识点画像。")
    with c2:
        pca_df = compute_pca_2d(factor_scores_df)
        if not pca_df.empty:
            pca_df["分层"] = assign_performance_band(pca_df["ability_mean"])
            pca_df["群组"] = simple_kmeans_labels(factor_scores_df, k=cluster_k)
            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="分层",
                symbol="群组",
                hover_data=["student_id", "ability_mean", "群组"],
                title="学生二维投影：PCA + 分层 + 聚类",
                color_discrete_sequence=[ACCENT_3, ACCENT_2, ACCENT],
            )
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        else:
            st.info("维度过少，无法绘制 PCA 投影。")

    d1, d2 = st.columns(2)
    with d1:
        if not item_df.empty and {"a", "b"}.issubset(item_df.columns):
            tmp = item_df.copy()
            tmp["难度标签"] = pd.cut(tmp["b"], bins=[-10, -1, 1, 10], labels=["偏易", "适中", "偏难"])
            fig = px.scatter(
                tmp,
                x="b",
                y="a",
                color="难度标签",
                size="样本量" if "样本量" in tmp.columns else None,
                hover_data=[c for c in ["qid", "正确率", "点二列相关", "高低组区分度"] if c in tmp.columns],
                title="题目参数地图：区分度 vs 难度",
            )
            st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        else:
            st.info("暂无题目参数地图。")
    with d2:
        if not item_df.empty and "a" in item_df.columns:
            top_df = item_df.sort_values("a", ascending=False).head(top_n)
            fig = px.bar(top_df.sort_values("a"), x="a", y="qid", orientation="h", title=f"高区分度题目 Top {top_n}", color_discrete_sequence=[ACCENT_4])
            st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        else:
            st.info("暂无高区分度题目榜单。")


def render_student_tab(run: Dict[str, Any]) -> None:
    irt = run["irt"]
    mirt = run["mirt"]
    prep = run["mirt_prepare"]
    students_enriched = irt["students_enriched"]
    factor_scores_df = mirt["student_factor_scores_df"]
    factor_meta_df = mirt["factor_meta_df"]
    question_mid_map_payload = json.loads(Path(prep["paths"]["question_mid_map_json"]).read_text(encoding="utf-8"))

    student_ids = [str(x.get("学生id")) for x in students_enriched]
    sid = st.selectbox("选择学生 ID", student_ids)
    report = build_student_knowledge_report(sid, students_enriched, factor_scores_df, factor_meta_df, question_mid_map_payload)
    stu = report["student"]
    mastery_df = report["mastery_df"].copy()
    weak_df = report["weak_df"].copy()
    strong_df = report["strong_df"].copy()
    detail_df = report["detail_df"].copy()
    overview = stu.get("统计概览", {})
    ability = stu.get("IRT能力", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("正确率", pretty_pct(overview.get("正确率")))
    c2.metric("平均得分率", pretty_pct(overview.get("平均得分率")))
    c3.metric("IRT Theta", "-" if ability.get("theta") is None else f"{safe_float(ability['theta']):.3f}")
    c4.metric("Theta SE", "-" if ability.get("theta_se") is None else f"{safe_float(ability['theta_se']):.3f}")

    left, right = st.columns([0.95, 1.05])
    with left:
        st.subheader("薄弱知识点雷达图")
        if mastery_df.empty:
            st.info("该学生暂无 MIRT 结果。")
        else:
            radar_df = mastery_df.sort_values("mastery").head(8)
            fig = make_closed_radar(radar_df["mid_name"].tolist(), radar_df["mastery"].fillna(0).tolist(), "掌握度")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("掌握度 vs 练习正确率")
        if mastery_df.empty:
            st.info("暂无学生知识点画像。")
        else:
            tmp = mastery_df.copy().sort_values("mastery").head(10)
            tmp["exercise_acc_pct"] = tmp["exercise_acc"].fillna(0) * 100
            fig = go.Figure()
            fig.add_trace(go.Bar(x=tmp["mastery"], y=tmp["mid_name"], orientation="h", name="掌握度", marker=dict(color=ACCENT)))
            fig.add_trace(go.Bar(x=tmp["exercise_acc_pct"], y=tmp["mid_name"], orientation="h", name="练习正确率", marker=dict(color=ACCENT_2)))
            fig.update_layout(barmode="group", title="学生薄弱维度进步卡")
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    d1, d2, d3 = st.columns([0.9, 0.9, 1.2])
    with d1:
        st.markdown("**最薄弱 Top 8**")
        st.dataframe(weak_df[["mid_name", "mastery", "exercise_acc", "n_items", "exercise_n"]], use_container_width=True, height=300)
    with d2:
        st.markdown("**最优势 Top 8**")
        st.dataframe(strong_df[["mid_name", "mastery", "exercise_acc", "n_items", "exercise_n"]], use_container_width=True, height=300)
    with d3:
        if not mastery_df.empty:
            tmp = mastery_df.copy()
            tmp["等级"] = pd.cut(tmp["mastery"], bins=[-0.1, 40, 60, 80, 101], labels=["重点补救", "待提升", "基本稳定", "优势"])
            fig = px.bar(tmp.groupby("等级", as_index=False).size(), x="等级", y="size", title="该学生知识点等级分布", color="等级")
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        else:
            st.info("暂无等级分布。")

    st.subheader("知识点掌握度全景")
    if mastery_df.empty:
        st.info("暂无掌握度全景。")
    else:
        tmp = mastery_df.copy().sort_values("mastery")
        tmp["等级"] = pd.cut(tmp["mastery"], bins=[-0.1, 40, 60, 80, 101], labels=["重点补救", "待提升", "基本稳定", "优势"])
        fig = px.bar(
            tmp,
            x="mastery",
            y="mid_name",
            orientation="h",
            color="等级",
            hover_data=["score", "exercise_acc", "n_items", "exercise_n"],
            title="知识点掌握度排序",
        )
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)
        st.dataframe(tmp, use_container_width=True, height=340)

    st.subheader("学生作答明细")
    if detail_df.empty:
        st.info("暂无作答明细。")
    else:
        st.dataframe(detail_df, use_container_width=True, height=280)
        df_download_button(mastery_df, f"下载学生 {sid} 知识点报告 CSV", f"student_{sid}_mastery.csv")


def render_item_tab(run: Dict[str, Any], top_n: int) -> None:
    item_df = get_item_merged_df(run)
    if item_df.empty:
        st.info("暂无题目参数。")
        return

    st.subheader("题目诊断面板")
    a1, a2 = st.columns([1.05, 0.95])
    with a1:
        if {"a", "b"}.issubset(item_df.columns):
            tmp = item_df.copy()
            tmp["难度标签"] = pd.cut(tmp["b"], bins=[-10, -1, 1, 10], labels=["偏易", "适中", "偏难"])
            fig = px.scatter(
                tmp,
                x="b",
                y="a",
                color="难度标签",
                size="样本量" if "样本量" in tmp.columns else None,
                hover_data=[c for c in ["qid", "正确率", "点二列相关", "高低组区分度"] if c in tmp.columns],
                title="题目参数地图",
            )
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    with a2:
        if "a" in item_df.columns:
            top_a = item_df.sort_values("a", ascending=False).head(top_n)
            fig = px.bar(top_a.sort_values("a"), x="a", y="qid", orientation="h", title=f"区分度 Top {top_n}", color_discrete_sequence=[ACCENT_4])
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    selectable = item_df["qid"].astype(str).tolist() if "qid" in item_df.columns else []
    qid = st.selectbox("查看单题诊断曲线", selectable)
    curve_df, row = build_item_curve_df(item_df, qid)
    if curve_df.empty or row is None:
        st.info("该题暂无可视化曲线。")
    else:
        b1, b2, b3 = st.columns([1.0, 1.0, 0.9])
        with b1:
            fig = px.line(curve_df, x="theta", y="ICC", title=f"题目 {qid} 的 ICC 曲线", color_discrete_sequence=[ACCENT])
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        with b2:
            fig = px.line(curve_df, x="theta", y="IIF", title=f"题目 {qid} 的 IIF 曲线", color_discrete_sequence=[ACCENT_2])
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        with b3:
            st.markdown('<div class="note-card">', unsafe_allow_html=True)
            st.metric("a", f"{safe_float(row.get('a')):.3f}")
            st.metric("b", f"{safe_float(row.get('b')):.3f}")
            if "正确率" in row.index:
                st.metric("正确率", pretty_pct(row.get("正确率")))
            if "点二列相关" in row.index:
                val = row.get("点二列相关")
                st.metric("点二列相关", "-" if pd.isna(val) else f"{float(val):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if "b" in item_df.columns:
            fig = px.box(item_df, y="b", points="all", title="题目难度 b 箱线图", color_discrete_sequence=[ACCENT_3])
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with c2:
        if "a" in item_df.columns:
            fig = px.violin(item_df, y="a", box=True, points="all", title="题目区分度 a 小提琴图", color_discrete_sequence=[ACCENT])
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    st.subheader("题目参数与 CTT 指标表")
    st.dataframe(item_df, use_container_width=True, height=380)


def render_cohort_tab(run: Dict[str, Any], cluster_k: int) -> None:
    theta_df = run["irt"]["theta_df"].copy()
    factor_scores_df = run["mirt"]["student_factor_scores_df"].copy()
    factor_meta_df = run["mirt"]["factor_meta_df"].copy()
    cohort_df = build_cohort_factor_df(factor_scores_df, factor_meta_df)
    corr = rename_factor_labels(build_factor_corr_df(factor_scores_df), factor_meta_df)

    st.subheader("群体画像")
    a1, a2 = st.columns(2)
    with a1:
        if not factor_scores_df.empty:
            melted = factor_scores_df.melt(id_vars=["student_id"], value_vars=get_score_columns(factor_scores_df), var_name="factor", value_name="score")
            name_map = dict(zip(factor_meta_df["factor_name"].astype(str), factor_meta_df["mid_name"].astype(str))) if not factor_meta_df.empty else {}
            melted["维度"] = melted["factor"].astype(str).map(lambda x: name_map.get(x, x))
            fig = px.box(melted, x="维度", y="score", title="各知识点维度能力分布", color="维度")
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        else:
            st.info("暂无维度分布。")
    with a2:
        if not corr.empty:
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="维度相关热力图")
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        else:
            st.info("至少需要两个维度才能绘制相关图。")

    b1, b2 = st.columns([1.05, 0.95])
    with b1:
        pca_df = compute_pca_2d(factor_scores_df)
        if not pca_df.empty:
            pca_df["分层"] = assign_performance_band(pca_df["ability_mean"])
            pca_df["群组"] = simple_kmeans_labels(factor_scores_df, k=cluster_k)
            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2",
                color="群组",
                symbol="分层",
                hover_data=["student_id", "ability_mean", "分层"],
                title="群体聚类散点图",
            )
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)
        else:
            st.info("暂无聚类散点。")
    with b2:
        if not theta_df.empty and "theta" in theta_df.columns:
            theta_df["分层"] = assign_performance_band(theta_df["theta"])
            fig = px.histogram(theta_df, x="theta", color="分层", title="学生能力分层直方图", nbins=28)
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)
        else:
            st.info("暂无学生分层直方图。")

    st.subheader("群体知识点统计表")
    if cohort_df.empty:
        st.info("暂无群体知识点统计。")
    else:
        st.dataframe(cohort_df, use_container_width=True, height=320)


def render_fit_tab(run: Dict[str, Any]) -> None:
    irt = run["irt"]
    mirt = run["mirt"]
    prep = run["mirt_prepare"]

    st.subheader("拟合过程与诊断")
    a1, a2 = st.columns([1.05, 0.95])
    with a1:
        history = irt["fit_result"].get("history", [])
        if history:
            loss_df = pd.DataFrame({"iter": range(1, len(history) + 1), "loss": history})
            fig = px.line(loss_df, x="iter", y="loss", markers=True, title="IRT 损失下降曲线", color_discrete_sequence=[ACCENT])
            st.plotly_chart(style_fig(fig, 350), use_container_width=True)
        else:
            st.info("暂无 IRT 迭代历史。")
    with a2:
        st.markdown("**Python 近似 MIRT 模型摘要**")
        st.json(mirt.get("model_summary", {}))

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**MIRT 预处理摘要**")
        st.json(prep.get("summary", {}))
    with b2:
        st.markdown("**过滤日志**")
        st.json(prep.get("filter_log", {}))

    st.subheader("拟合日志")
    st.code("\n".join(mirt.get("console_lines", [])) or "无日志", language="text")


def render_data_tab(run: Dict[str, Any]) -> None:
    irt = run["irt"]
    mirt = run["mirt"]
    prep = run["mirt_prepare"]
    tables = {
        "IRT 长表": irt["long_df"],
        "学生汇总": irt["student_summary_df"],
        "题目 CTT 指标": irt["item_stats_df"],
        "IRT 题目参数": irt["item_param_df"],
        "IRT 学生能力": irt["theta_df"],
        "MIRT 响应矩阵": prep["response_matrix_df"],
        "MIRT 题目映射": prep["item_meta_df"],
        "MIRT 因子映射": mirt["factor_meta_df"],
        "MIRT 学生因子分数": mirt["student_factor_scores_df"],
        "MIRT 题目参数": mirt["item_params_df"],
    }
    chosen = st.selectbox("选择数据表", list(tables.keys()))
    st.dataframe(tables[chosen], use_container_width=True, height=500)


def render_export_tab(run: Dict[str, Any]) -> None:
    irt = run["irt"]
    mirt = run["mirt"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        json_download_button(irt["students_enriched"], "下载增强学生 JSON", "students_enriched.json")
    with c2:
        json_download_button(irt["question_bank_enriched"], "下载增强题库 JSON", "question_bank_enriched.json")
    with c3:
        df_download_button(irt["theta_df"], "下载 IRT 学生能力 CSV", "irt_theta.csv")
    with c4:
        df_download_button(irt["item_param_df"], "下载 IRT 题目参数 CSV", "irt_item_params.csv")

    d1, d2 = st.columns(2)
    with d1:
        df_download_button(mirt["student_factor_scores_df"], "下载 MIRT 学生因子分数 CSV", "student_factor_scores.csv")
    with d2:
        df_download_button(mirt["item_params_df"], "下载 MIRT 题目参数 CSV", "mirt_item_params.csv")

    st.markdown("**运行目录**")
    st.code(run["work_dir"], language="text")
    st.markdown('<div class="footer-note">发布到 GitHub 时，建议把这个文件命名为 <code>streamlit_app.py</code> 或 <code>app.py</code>，并补一个 requirements.txt。</div>', unsafe_allow_html=True)


def main() -> None:
    inject_css()
    render_header()

    if "latest_run" not in st.session_state:
        st.session_state.latest_run = None

    inputs = render_sidebar()
    run = run_pipeline_ui(inputs)
    if not run:
        st.info("上传 3 个文件并点击“开始一键运行”。")
        return

    render_summary_cards(run)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "总览仪表盘",
        "学生学情",
        "题目诊断",
        "群体画像",
        "拟合过程",
        "中间数据 / 导出",
    ])

    with tab1:
        render_overview_tab(run, inputs["cluster_k"], inputs["top_n"])
    with tab2:
        render_student_tab(run)
    with tab3:
        render_item_tab(run, inputs["top_n"])
    with tab4:
        render_cohort_tab(run, inputs["cluster_k"])
    with tab5:
        render_fit_tab(run)
    with tab6:
        render_data_tab(run)
        st.divider()
        render_export_tab(run)


if __name__ == "__main__":
    main()
