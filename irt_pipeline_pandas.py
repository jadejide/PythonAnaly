from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-8
MAX_EXP = 35.0


@dataclass
class IRTConfig:
    min_item_responses: int = 30
    min_student_responses: int = 10
    max_iters: int = 120
    learning_rate_theta: float = 0.01
    learning_rate_item: float = 0.01
    l2_theta: float = 0.01
    l2_a: float = 0.01
    l2_b: float = 0.01
    convergence_tol: float = 1e-4
    theta_clip: Tuple[float, float] = (-4.0, 4.0)
    a_clip: Tuple[float, float] = (0.2, 3.0)
    b_clip: Tuple[float, float] = (-4.0, 4.0)


# =========================
# 基础工具
# =========================
def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default



def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))



def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -MAX_EXP, MAX_EXP)
    return 1.0 / (1.0 + np.exp(-x))



def mean_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.mean()
    return None if pd.isna(value) else float(value)



def extract_question_knowledge(qobj: Dict[str, Any]) -> List[str]:
    fields: List[str] = []
    for key in ["knowledges", "bnu_knowledges", "xkb_knowledges"]:
        value = qobj.get(key)
        if isinstance(value, list):
            fields.extend([str(x) for x in value if x not in (None, "")])
    return list(dict.fromkeys(fields))


# =========================
# 读取 / 扁平化
# =========================
def load_json_from_path(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def build_question_df(question_bank_raw: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for q in question_bank_raw:
        rows.append(
            {
                "qid": q.get("id"),
                "stem": q.get("stem", ""),
                "question_obj": q,
                "question_knowledges": extract_question_knowledge(q),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["qid", "stem", "question_obj", "question_knowledges"])
    df["qid"] = df["qid"].astype(str)
    return df



def build_long_df(
    students_raw: List[Dict[str, Any]],
    question_df: pd.DataFrame,
    id_mapping: Dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_mapping = id_mapping or {}
    question_knowledge_map = (
        question_df.set_index("qid")["question_knowledges"].to_dict() if not question_df.empty else {}
    )

    student_rows: List[Dict[str, Any]] = []
    response_rows: List[Dict[str, Any]] = []

    for stu in students_raw:
        sid = str(stu.get("学生id"))
        student_rows.append({"sid": sid, "student_obj": stu})

        for idx, rec in enumerate(stu.get("详细记录", []) or []):
            raw_qid = str(rec.get("题目id"))
            qid = str(id_mapping.get(raw_qid, raw_qid))
            total = float(rec.get("总分", 0) or 0)
            score = float(rec.get("得分", 0) or 0)
            score_rate = safe_div(score, total, 0.0)
            is_correct = int(abs(score - total) < 1e-9 and total > 0)
            knowledges = rec.get("智学知识点", []) or question_knowledge_map.get(qid, []) or []

            response_rows.append(
                {
                    "sid": sid,
                    "record_idx": idx,
                    "raw_qid": raw_qid,
                    "qid": qid,
                    "score": score,
                    "total": total,
                    "score_rate": score_rate,
                    "is_correct": is_correct,
                    "knowledges": knowledges,
                }
            )

    students_df = pd.DataFrame(student_rows)
    long_df = pd.DataFrame(response_rows)

    if long_df.empty:
        long_df = pd.DataFrame(
            columns=[
                "sid",
                "record_idx",
                "raw_qid",
                "qid",
                "score",
                "total",
                "score_rate",
                "is_correct",
                "knowledges",
            ]
        )
    else:
        long_df["sid"] = long_df["sid"].astype(str)
        long_df["qid"] = long_df["qid"].astype(str)
        long_df["record_idx"] = long_df["record_idx"].astype(int)

    return students_df, long_df


# =========================
# pandas 驱动的统计
# =========================
def build_student_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(
            columns=["sid", "正确题数", "错误题数", "正确率", "平均得分率", "作答数"]
        )

    summary = (
        long_df.groupby("sid", as_index=False)
        .agg(
            作答数=("is_correct", "size"),
            正确题数=("is_correct", "sum"),
            正确率=("is_correct", "mean"),
            平均得分率=("score_rate", "mean"),
        )
        .assign(错误题数=lambda df: df["作答数"] - df["正确题数"])
    )

    for col in ["正确率", "平均得分率"]:
        summary[col] = summary[col].round(6)
    return summary[["sid", "正确题数", "错误题数", "正确率", "平均得分率", "作答数"]]



def compute_high_low_discrimination(group: pd.DataFrame, ratio: float = 0.27) -> float | None:
    if len(group) < 4:
        return None
    sorted_group = group.sort_values("student_avg_score_rate")
    g = max(1, int(round(len(sorted_group) * ratio)))
    low = sorted_group.head(g)["is_correct"]
    high = sorted_group.tail(g)["is_correct"]
    if low.empty or high.empty:
        return None
    return float(high.mean() - low.mean())



def build_item_ctt_stats(long_df: pd.DataFrame, question_df: pd.DataFrame) -> pd.DataFrame:
    base_qids = question_df[["qid"]].drop_duplicates() if not question_df.empty else pd.DataFrame(columns=["qid"])
    if long_df.empty:
        empty_stats = base_qids.copy()
        for col in ["作答人数", "满分人数", "正确率", "平均得分率", "点二列相关", "高低组区分度"]:
            empty_stats[col] = np.nan
        empty_stats[["作答人数", "满分人数"]] = empty_stats[["作答人数", "满分人数"]].fillna(0).astype(int)
        return empty_stats

    student_totals = (
        long_df.groupby("sid", as_index=False)
        .agg(student_total_correct=("is_correct", "sum"), student_n_items=("is_correct", "size"), student_avg_score_rate=("score_rate", "mean"))
    )

    df = long_df.merge(student_totals, on="sid", how="left")
    df["total_correct_excl_item"] = np.where(
        df["student_n_items"] > 1,
        (df["student_total_correct"] - df["is_correct"]) / (df["student_n_items"] - 1),
        0.0,
    )

    item_stats = (
        df.groupby("qid", as_index=False)
        .agg(
            作答人数=("is_correct", "size"),
            满分人数=("is_correct", "sum"),
            正确率=("is_correct", "mean"),
            平均得分率=("score_rate", "mean"),
        )
    )

    pbr = (
        df.groupby("qid")
        .apply(
            lambda g: g["is_correct"].corr(g["total_correct_excl_item"])
            if len(g) >= 2 and g["is_correct"].nunique() > 1 and g["total_correct_excl_item"].nunique() > 1
            else np.nan
        )
        .rename("点二列相关")
        .reset_index()
    )

    disc = (
        df.groupby("qid")
        .apply(compute_high_low_discrimination)
        .rename("高低组区分度")
        .reset_index()
    )

    item_stats = item_stats.merge(pbr, on="qid", how="left").merge(disc, on="qid", how="left")
    item_stats = base_qids.merge(item_stats, on="qid", how="left")

    item_stats["作答人数"] = item_stats["作答人数"].fillna(0).astype(int)
    item_stats["满分人数"] = item_stats["满分人数"].fillna(0).astype(int)
    for col in ["正确率", "平均得分率", "点二列相关", "高低组区分度"]:
        item_stats[col] = item_stats[col].round(6)
    return item_stats


# =========================
# IRT 矩阵与拟合
# =========================
def build_irt_matrix(
    long_df: pd.DataFrame,
    question_df: pd.DataFrame,
    config: IRTConfig,
) -> tuple[np.ndarray, list[str], list[str]]:
    if long_df.empty or question_df.empty:
        return np.empty((0, 0)), [], []

    valid_qids = set(question_df["qid"].astype(str).tolist())
    df = long_df[long_df["qid"].isin(valid_qids)].copy()
    if df.empty:
        return np.empty((0, 0)), [], []

    item_counts = df.groupby("qid")["sid"].size()
    candidate_qids = item_counts[item_counts >= config.min_item_responses].index.tolist()
    df = df[df["qid"].isin(candidate_qids)].copy()
    if df.empty:
        return np.empty((0, 0)), [], []

    student_counts = df.groupby("sid")["qid"].size()
    candidate_sids = student_counts[student_counts >= config.min_student_responses].index.tolist()
    df = df[df["sid"].isin(candidate_sids)].copy()
    if df.empty:
        return np.empty((0, 0)), [], []

    matrix_df = (
        df.pivot_table(index="sid", columns="qid", values="is_correct", aggfunc="first")
        .sort_index()
        .sort_index(axis=1)
    )

    if matrix_df.empty:
        return np.empty((0, 0)), [], []

    item_p = matrix_df.mean(axis=0, skipna=True)
    keep_cols = item_p[(item_p > 0.01) & (item_p < 0.99) & (matrix_df.count(axis=0) >= config.min_item_responses)].index
    matrix_df = matrix_df.loc[:, keep_cols]
    if matrix_df.empty:
        return np.empty((0, 0)), [], []

    keep_rows = matrix_df.count(axis=1) >= config.min_student_responses
    matrix_df = matrix_df.loc[keep_rows]
    if matrix_df.empty:
        return np.empty((0, 0)), [], []

    return matrix_df.to_numpy(dtype=float), matrix_df.index.astype(str).tolist(), matrix_df.columns.astype(str).tolist()



def normalize_theta(theta: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(theta.mean())
    sd = float(theta.std())
    if sd < EPS:
        sd = 1.0
    return (theta - mu) / sd, mu, sd



def fit_2pl_jml(X: np.ndarray, config: IRTConfig) -> Dict[str, Any]:
    if X.size == 0:
        return {
            "theta": np.array([]),
            "a": np.array([]),
            "b": np.array([]),
            "converged": False,
            "n_students": 0,
            "n_items": 0,
            "history": [],
        }

    mask = ~np.isnan(X)
    N, M = X.shape

    row_mean = np.nanmean(X, axis=1)
    col_mean = np.nanmean(X, axis=0)

    theta = np.clip(logit(np.nan_to_num(row_mean, nan=0.5)), -3.0, 3.0).astype(float)
    a = np.ones(M, dtype=float)
    b = np.clip(-logit(np.nan_to_num(col_mean, nan=0.5)), -3.0, 3.0).astype(float)

    prev_loss = None
    converged = False
    history: list[float] = []

    for _ in range(config.max_iters):
        diff = theta[:, None] - b[None, :]
        linear = a[None, :] * diff
        P = sigmoid(linear)
        residual = np.where(mask, X - P, 0.0)

        theta_grad = residual @ a - config.l2_theta * theta
        theta = theta + config.learning_rate_theta * theta_grad
        theta = np.clip(theta, *config.theta_clip)

        diff = theta[:, None] - b[None, :]
        linear = a[None, :] * diff
        P = sigmoid(linear)
        residual = np.where(mask, X - P, 0.0)

        grad_a = np.sum(residual * diff, axis=0) - config.l2_a * (a - 1.0)
        grad_b = np.sum(-residual * a[None, :], axis=0) - config.l2_b * b

        a = np.clip(a + config.learning_rate_item * grad_a, *config.a_clip)
        b = np.clip(b + config.learning_rate_item * grad_b, *config.b_clip)

        theta, mu, sd = normalize_theta(theta)
        b = np.clip((b - mu) / sd, *config.b_clip)
        a = np.clip(a * sd, *config.a_clip)

        diff = theta[:, None] - b[None, :]
        P = np.clip(sigmoid(a[None, :] * diff), EPS, 1.0 - EPS)
        neg_loglik = -np.where(mask, X * np.log(P) + (1 - X) * np.log(1 - P), 0.0).sum()
        reg = (
            0.5 * config.l2_theta * np.sum(theta ** 2)
            + 0.5 * config.l2_a * np.sum((a - 1.0) ** 2)
            + 0.5 * config.l2_b * np.sum(b ** 2)
        )
        loss = float(neg_loglik + reg)
        history.append(loss)

        if prev_loss is not None and abs(prev_loss - loss) < config.convergence_tol:
            converged = True
            break
        prev_loss = loss

    P = sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    theta_info = np.where(mask, (a[None, :] ** 2) * P * (1 - P), 0.0).sum(axis=1)
    theta_se = np.where(theta_info > EPS, 1.0 / np.sqrt(theta_info), np.nan)

    item_info = np.where(mask, (a[None, :] ** 2) * P * (1 - P), 0.0).sum(axis=0)
    b_se = np.where(item_info > EPS, 1.0 / np.sqrt(item_info), np.nan)
    item_sample_sizes = mask.sum(axis=0).astype(int)

    return {
        "theta": theta,
        "theta_se": theta_se,
        "a": a,
        "b": b,
        "b_se": b_se,
        "item_sample_sizes": item_sample_sizes,
        "converged": converged,
        "n_students": N,
        "n_items": M,
        "history": history,
        "prob": P,
    }


# =========================
# 结果回填
# =========================
def build_irt_maps(
    student_ids: Iterable[str],
    item_ids: Iterable[str],
    fit_result: Dict[str, Any],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    student_map: Dict[str, Dict[str, Any]] = {}
    item_map: Dict[str, Dict[str, Any]] = {}

    theta = fit_result.get("theta", np.array([]))
    theta_se = fit_result.get("theta_se", np.array([]))
    a = fit_result.get("a", np.array([]))
    b = fit_result.get("b", np.array([]))
    b_se = fit_result.get("b_se", np.array([]))
    converged = fit_result.get("converged", False)
    item_sample_sizes = fit_result.get("item_sample_sizes", np.array([], dtype=int))

    for sid, theta_val, theta_se_val in zip(student_ids, theta, theta_se):
        student_map[str(sid)] = {
            "model": "2PL",
            "theta": round(float(theta_val), 6),
            "theta_se": None if np.isnan(theta_se_val) else round(float(theta_se_val), 6),
            "收敛状态": "success" if converged else "max_iter",
        }

    for idx, (qid, a_val, b_val, b_se_val) in enumerate(zip(item_ids, a, b, b_se)):
        sample_size = int(item_sample_sizes[idx]) if idx < len(item_sample_sizes) else 0
        item_map[str(qid)] = {
            "model": "2PL",
            "a": round(float(a_val), 6),
            "b": round(float(b_val), 6),
            "c": None,
            "se_a": None,
            "se_b": None if np.isnan(b_se_val) else round(float(b_se_val), 6),
            "样本量": sample_size,
            "收敛状态": "success" if converged else "max_iter",
        }

    return student_map, item_map



def enrich_question_bank(
    question_bank_raw: List[Dict[str, Any]],
    item_stats_df: pd.DataFrame,
    item_param_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    item_stats_map = item_stats_df.set_index("qid").to_dict(orient="index") if not item_stats_df.empty else {}
    enriched = deepcopy(question_bank_raw)

    for q in enriched:
        qid = str(q.get("id"))
        stats = item_stats_map.get(qid, {})
        q["统计指标"] = {
            "作答人数": int(stats.get("作答人数", 0) or 0),
            "满分人数": int(stats.get("满分人数", 0) or 0),
            "正确率": None if pd.isna(stats.get("正确率")) else float(stats.get("正确率")),
            "平均得分率": None if pd.isna(stats.get("平均得分率")) else float(stats.get("平均得分率")),
            "点二列相关": None if pd.isna(stats.get("点二列相关")) else float(stats.get("点二列相关")),
            "高低组区分度": None if pd.isna(stats.get("高低组区分度")) else float(stats.get("高低组区分度")),
        }
        q["IRT参数"] = item_param_map.get(
            qid,
            {
                "model": "2PL",
                "a": None,
                "b": None,
                "c": None,
                "se_a": None,
                "se_b": None,
                "样本量": q["统计指标"]["作答人数"],
                "收敛状态": "not_estimated",
            },
        )
    return enriched



def build_knowledge_profile(
    stu: Dict[str, Any],
    question_lookup: Dict[str, Dict[str, Any]],
    item_param_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    bucket: Dict[str, list[Dict[str, Any]]] = {}

    for rec in stu.get("详细记录", []) or []:
        qid = str(rec.get("题目id_标准")) if rec.get("题目id_标准") is not None else None
        qobj = question_lookup.get(qid, {}) if qid else {}
        knowledges = rec.get("智学知识点", []) or extract_question_knowledge(qobj)
        for kp in knowledges:
            bucket.setdefault(str(kp), []).append(rec)

    profiles: List[Dict[str, Any]] = []
    for kp, recs in bucket.items():
        corrects = [float(r.get("是否正确", 0) or 0) for r in recs]
        qids = [str(r.get("题目id_标准")) for r in recs if r.get("题目id_标准") is not None]
        a_list = [item_param_map[qid]["a"] for qid in qids if qid in item_param_map and item_param_map[qid].get("a") is not None]
        b_list = [item_param_map[qid]["b"] for qid in qids if qid in item_param_map and item_param_map[qid].get("b") is not None]

        profiles.append(
            {
                "知识点": kp,
                "作答数": len(recs),
                "正确率": round(float(np.mean(corrects)), 6) if corrects else None,
                "平均题目难度b": round(float(np.mean(b_list)), 6) if b_list else None,
                "平均题目区分度a": round(float(np.mean(a_list)), 6) if a_list else None,
            }
        )

    profiles.sort(key=lambda x: (-x["作答数"], x["知识点"]))
    return profiles



def enrich_students(
    students_raw: List[Dict[str, Any]],
    student_summary_df: pd.DataFrame,
    student_param_map: Dict[str, Dict[str, Any]],
    item_param_map: Dict[str, Dict[str, Any]],
    id_mapping: Dict[str, str] | None,
    question_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    id_mapping = id_mapping or {}
    summary_map = student_summary_df.set_index("sid").to_dict(orient="index") if not student_summary_df.empty else {}
    enriched = deepcopy(students_raw)

    for stu in enriched:
        sid = str(stu.get("学生id"))
        summary = summary_map.get(sid, {})
        stu["统计概览"] = {
            "正确题数": int(summary.get("正确题数", 0) or 0),
            "错误题数": int(summary.get("错误题数", 0) or 0),
            "正确率": float(summary.get("正确率", 0.0) or 0.0),
            "平均得分率": float(summary.get("平均得分率", 0.0) or 0.0),
        }
        stu["IRT能力"] = student_param_map.get(
            sid,
            {"model": "2PL", "theta": None, "theta_se": None, "收敛状态": "not_estimated"},
        )

        theta_val = stu["IRT能力"].get("theta")
        for rec in stu.get("详细记录", []) or []:
            raw_qid = str(rec.get("题目id"))
            qid = str(id_mapping.get(raw_qid, raw_qid))
            total = float(rec.get("总分", 0) or 0)
            score = float(rec.get("得分", 0) or 0)
            score_rate = safe_div(score, total, 0.0)
            is_correct = int(abs(score - total) < 1e-9 and total > 0)

            rec["题目id_标准"] = qid
            rec["得分率"] = round(score_rate, 6)
            rec["是否正确"] = is_correct

            item_param = item_param_map.get(qid)
            if item_param and theta_val is not None and item_param.get("a") is not None and item_param.get("b") is not None:
                pred = float(sigmoid(item_param["a"] * (theta_val - item_param["b"])))
                rec["IRT作答信息"] = {
                    "a": item_param["a"],
                    "b": item_param["b"],
                    "c": None,
                    "theta": theta_val,
                    "预测正确概率": round(pred, 6),
                    "能力差值_theta_minus_b": round(theta_val - item_param["b"], 6),
                }
            else:
                rec["IRT作答信息"] = {
                    "a": None,
                    "b": None,
                    "c": None,
                    "theta": theta_val,
                    "预测正确概率": None,
                    "能力差值_theta_minus_b": None,
                }

        stu["知识点画像"] = build_knowledge_profile(stu, question_lookup, item_param_map)

    return enriched


# =========================
# 前端图表辅助
# =========================
def icc_curve(theta_grid: np.ndarray, a: float, b: float) -> np.ndarray:
    return sigmoid(a * (theta_grid - b))



def iif_curve(theta_grid: np.ndarray, a: float, b: float) -> np.ndarray:
    p = icc_curve(theta_grid, a, b)
    return (a ** 2) * p * (1 - p)



def test_information_curve(theta_grid: np.ndarray, item_param_df: pd.DataFrame) -> np.ndarray:
    info = np.zeros_like(theta_grid, dtype=float)
    for _, row in item_param_df.iterrows():
        info += iif_curve(theta_grid, float(row["a"]), float(row["b"]))
    return info



def test_se_curve(theta_grid: np.ndarray, item_param_df: pd.DataFrame) -> np.ndarray:
    info = test_information_curve(theta_grid, item_param_df)
    return np.where(info > EPS, 1.0 / np.sqrt(info), np.nan)


# =========================
# 主流程
# =========================
def run_pipeline(
    students_raw: List[Dict[str, Any]],
    question_bank_raw: List[Dict[str, Any]],
    id_mapping: Dict[str, str] | None = None,
    config: IRTConfig | None = None,
) -> Dict[str, Any]:
    config = config or IRTConfig()

    question_df = build_question_df(question_bank_raw)
    students_df, long_df = build_long_df(students_raw, question_df, id_mapping=id_mapping)
    student_summary_df = build_student_summary(long_df)
    item_stats_df = build_item_ctt_stats(long_df, question_df)

    X, student_ids, item_ids = build_irt_matrix(long_df, question_df, config)
    fit_result = fit_2pl_jml(X, config)
    student_param_map, item_param_map = build_irt_maps(student_ids, item_ids, fit_result)

    question_lookup = {str(q.get("id")): q for q in question_bank_raw}
    enriched_question_bank = enrich_question_bank(question_bank_raw, item_stats_df, item_param_map)
    enriched_students = enrich_students(
        students_raw,
        student_summary_df,
        student_param_map,
        item_param_map,
        id_mapping=id_mapping,
        question_lookup=question_lookup,
    )

    item_param_df = pd.DataFrame(
        [{"qid": qid, **params} for qid, params in item_param_map.items() if params.get("a") is not None and params.get("b") is not None]
    )
    theta_df = pd.DataFrame(
        [{"sid": sid, **params} for sid, params in student_param_map.items() if params.get("theta") is not None]
    )

    return {
        "students_enriched": enriched_students,
        "question_bank_enriched": enriched_question_bank,
        "students_df": students_df,
        "long_df": long_df,
        "student_summary_df": student_summary_df,
        "item_stats_df": item_stats_df,
        "item_param_df": item_param_df,
        "theta_df": theta_df,
        "fit_result": fit_result,
        "metrics": {
            "students_total": int(long_df["sid"].nunique()) if not long_df.empty else 0,
            "items_total": int(question_df["qid"].nunique()) if not question_df.empty else 0,
            "responses_total": int(len(long_df)),
            "irt_students": int(fit_result.get("n_students", 0)),
            "irt_items": int(fit_result.get("n_items", 0)),
            "converged": bool(fit_result.get("converged", False)),
        },
    }



def save_json(data: Any, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
