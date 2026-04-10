from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from irt_pipeline_pandas import IRTConfig, run_pipeline


@dataclass
class MIRTPrepConfig:
    min_resp_per_item: int = 30
    min_resp_per_student: int = 10
    min_items_per_factor: int = 10


# -------------------------
# IO helpers
# -------------------------
def _load_json_auto(source: str | os.PathLike | bytes | io.BytesIO) -> Any:
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    if isinstance(source, bytes):
        return json.loads(source.decode("utf-8"))
    if hasattr(source, "read"):
        raw = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        if isinstance(raw, bytes):
            return json.loads(raw.decode("utf-8"))
        return json.loads(raw)
    raise TypeError(f"Unsupported JSON source type: {type(source)}")


def _save_json(data: Any, path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _copy_or_write_uploaded(file_obj: Any, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(file_obj, (str, os.PathLike)):
        shutil.copy2(str(file_obj), str(out_path))
        return out_path

    data = file_obj.getbuffer() if hasattr(file_obj, "getbuffer") else file_obj.read()
    with open(out_path, "wb") as f:
        f.write(data)
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    return out_path


# -------------------------
# MIRT preprocess
# -------------------------
def get_question_id_from_record(rec: Dict[str, Any]) -> str:
    return str(rec.get("题目id_标准", rec.get("题目id")))


def get_is_correct(rec: Dict[str, Any]) -> int:
    total = float(rec.get("总分", 0) or 0)
    score = float(rec.get("得分", 0) or 0)
    if total <= 0:
        return 0
    return 1 if abs(score - total) < 1e-9 else 0


def load_taxonomy_mapping(excel_path: str | os.PathLike) -> tuple[Dict[str, str], Dict[str, str]]:
    df = pd.read_excel(excel_path)
    required_cols = ["二级知识点", "二级知识点编号", "四级知识点"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"知识体系 Excel 缺少列: {missing}")

    xkb4_to_mid: Dict[str, str] = {}
    mid_id_to_name: Dict[str, str] = {}
    for _, row in df.iterrows():
        mid_name = str(row["二级知识点"]).strip()
        mid_id = str(row["二级知识点编号"]).strip()
        xkb4 = str(row["四级知识点"]).strip()
        if not xkb4 or xkb4 == "nan":
            continue
        xkb4_to_mid[xkb4] = mid_id
        mid_id_to_name[mid_id] = mid_name
    return xkb4_to_mid, mid_id_to_name


def build_question_mid_map(question_bank: List[Dict[str, Any]], xkb4_to_mid: Dict[str, str]) -> Dict[str, List[str]]:
    question_mid_map: Dict[str, List[str]] = {}
    for q in question_bank:
        qid = str(q.get("id"))
        xkb_list = q.get("xkb_knowledges", []) or q.get("新课标知识点", []) or []
        mids: List[str] = []
        for xkb in xkb_list:
            key = str(xkb).strip()
            if key in xkb4_to_mid:
                mids.append(xkb4_to_mid[key])
        question_mid_map[qid] = list(dict.fromkeys(mids))
    return question_mid_map


def build_response_long(students: List[Dict[str, Any]], valid_qids: set[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for stu in students:
        sid = str(stu.get("学生id"))
        for rec in stu.get("详细记录", []) or []:
            qid = get_question_id_from_record(rec)
            if qid not in valid_qids:
                continue
            rows.append({"student_id": sid, "item_id": qid, "score01": get_is_correct(rec)})
    return pd.DataFrame(rows)


def build_response_matrix(response_long: pd.DataFrame) -> pd.DataFrame:
    mat = response_long.pivot_table(index="student_id", columns="item_id", values="score01", aggfunc="max")
    return mat.sort_index().sort_index(axis=1)


def rename_items_to_safe_names(response_matrix: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    old_item_ids = list(response_matrix.columns)
    safe_names = [f"Item_{i+1:05d}" for i in range(len(old_item_ids))]
    rename_map = dict(zip(old_item_ids, safe_names))
    reverse_map = dict(zip(safe_names, old_item_ids))
    return response_matrix.rename(columns=rename_map), rename_map, reverse_map


def stabilize_matrix(
    response_matrix: pd.DataFrame,
    reverse_map: Dict[str, str],
    question_mid_map: Dict[str, List[str]],
    cfg: MIRTPrepConfig,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    current = response_matrix.copy()
    filter_log: Dict[str, Any] = {"iterations": []}
    changed = True
    while changed:
        changed = False
        step_info: Dict[str, Any] = {}

        all_na_rows = current.isna().all(axis=1)
        all_na_cols = current.isna().all(axis=0)
        if all_na_rows.any():
            current = current.loc[~all_na_rows, :]
            changed = True
        if all_na_cols.any():
            current = current.loc[:, ~all_na_cols]
            changed = True
        step_info["drop_all_na_rows"] = int(all_na_rows.sum())
        step_info["drop_all_na_cols"] = int(all_na_cols.sum())

        constant_cols = []
        for c in current.columns:
            vals = current[c].dropna().unique()
            if len(vals) <= 1:
                constant_cols.append(c)
        if constant_cols:
            current = current.drop(columns=constant_cols)
            changed = True
        step_info["drop_constant_cols"] = len(constant_cols)

        item_non_missing = current.notna().sum(axis=0)
        bad_items = item_non_missing[item_non_missing < cfg.min_resp_per_item].index.tolist()
        if bad_items:
            current = current.drop(columns=bad_items)
            changed = True
        step_info["drop_low_resp_items"] = len(bad_items)

        stu_non_missing = current.notna().sum(axis=1)
        bad_students = stu_non_missing[stu_non_missing < cfg.min_resp_per_student].index.tolist()
        if bad_students:
            current = current.drop(index=bad_students)
            changed = True
        step_info["drop_low_resp_students"] = len(bad_students)

        mid_to_items: Dict[str, List[str]] = defaultdict(list)
        for safe_item in current.columns:
            qid = reverse_map[safe_item]
            for mid_id in question_mid_map.get(qid, []):
                mid_to_items[mid_id].append(safe_item)

        kept_mid_ids = {mid_id for mid_id, items in mid_to_items.items() if len(items) >= cfg.min_items_per_factor}
        kept_items: set[str] = set()
        for mid_id in kept_mid_ids:
            kept_items.update(mid_to_items[mid_id])

        bad_factor_items = [c for c in current.columns if c not in kept_items]
        if bad_factor_items:
            current = current.drop(columns=bad_factor_items)
            changed = True
        step_info["kept_mid_count"] = len(kept_mid_ids)
        step_info["drop_items_not_in_kept_factors"] = len(bad_factor_items)
        step_info["n_students_after"] = int(current.shape[0])
        step_info["n_items_after"] = int(current.shape[1])
        filter_log["iterations"].append(step_info)

    return current, filter_log


def build_mirt_model_text(
    final_matrix: pd.DataFrame,
    reverse_map: Dict[str, str],
    question_mid_map: Dict[str, List[str]],
    mid_id_to_name: Dict[str, str],
    cfg: MIRTPrepConfig,
) -> tuple[str, List[Dict[str, Any]], List[str]]:
    safe_item_names = list(final_matrix.columns)
    item_pos_map = {safe_item: idx + 1 for idx, safe_item in enumerate(safe_item_names)}
    mid_to_item_positions: Dict[str, List[int]] = defaultdict(list)
    for safe_item in safe_item_names:
        qid = reverse_map[safe_item]
        for mid_id in question_mid_map.get(qid, []):
            mid_to_item_positions[mid_id].append(item_pos_map[safe_item])

    filtered_mid_to_positions = {
        mid_id: positions
        for mid_id, positions in mid_to_item_positions.items()
        if len(positions) >= cfg.min_items_per_factor
    }

    factor_name_map: Dict[str, str] = {}
    model_lines: List[str] = []
    factor_meta: List[Dict[str, Any]] = []
    kept_positions: set[int] = set()

    for idx, mid_id in enumerate(sorted(filtered_mid_to_positions.keys()), start=1):
        factor_name = f"F{idx}"
        factor_name_map[mid_id] = factor_name
        positions = filtered_mid_to_positions[mid_id]
        pos_part = ",".join(str(p) for p in positions)
        model_lines.append(f"{factor_name} = {pos_part}")
        factor_meta.append({
            "mid_id": mid_id,
            "mid_name": mid_id_to_name.get(mid_id, mid_id),
            "factor_name": factor_name,
            "n_items": len(positions),
        })
        kept_positions.update(positions)

    model_text = "\n".join(model_lines) + "\n"
    kept_safe_items = [safe_item_names[p - 1] for p in sorted(kept_positions)]
    return model_text, factor_meta, kept_safe_items


def prepare_mirt_inputs(
    students_raw: List[Dict[str, Any]],
    question_bank_raw: List[Dict[str, Any]],
    taxonomy_excel_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    cfg: MIRTPrepConfig | None = None,
) -> Dict[str, Any]:
    cfg = cfg or MIRTPrepConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xkb4_to_mid, mid_id_to_name = load_taxonomy_mapping(taxonomy_excel_path)
    question_mid_map = build_question_mid_map(question_bank_raw, xkb4_to_mid)
    valid_qids = {qid for qid, mids in question_mid_map.items() if mids}
    response_long = build_response_long(students_raw, valid_qids)
    response_matrix = build_response_matrix(response_long)
    response_matrix, rename_map, reverse_map = rename_items_to_safe_names(response_matrix)
    final_matrix, filter_log = stabilize_matrix(response_matrix, reverse_map, question_mid_map, cfg)
    model_text, factor_meta, kept_safe_items = build_mirt_model_text(final_matrix, reverse_map, question_mid_map, mid_id_to_name, cfg)
    final_matrix = final_matrix.loc[:, kept_safe_items]
    final_item_pos_map = {safe_item: idx + 1 for idx, safe_item in enumerate(final_matrix.columns)}

    item_meta_rows = []
    for safe_item in final_matrix.columns:
        qid = reverse_map[safe_item]
        item_meta_rows.append({
            "item_position": final_item_pos_map[safe_item],
            "safe_item_id": safe_item,
            "original_item_id": qid,
            "mid_ids": "|".join(question_mid_map.get(qid, [])),
        })
    item_meta = pd.DataFrame(item_meta_rows)

    paths = {
        "response_matrix_csv": out_dir / "response_matrix.csv",
        "item_meta_csv": out_dir / "item_meta.csv",
        "mirt_model_txt": out_dir / "mirt_model.txt",
        "question_mid_map_json": out_dir / "question_mid_map.json",
        "item_name_map_json": out_dir / "item_name_map.json",
        "filter_log_json": out_dir / "filter_log.json",
        "factor_meta_csv": out_dir / "factor_meta.csv",
    }
    final_matrix.to_csv(paths["response_matrix_csv"], encoding="utf-8-sig")
    item_meta.to_csv(paths["item_meta_csv"], index=False, encoding="utf-8-sig")
    paths["mirt_model_txt"].write_text(model_text, encoding="utf-8")
    _save_json({"question_mid_map": question_mid_map, "factor_meta": factor_meta}, paths["question_mid_map_json"])
    _save_json({"original_to_safe": rename_map, "safe_to_original": reverse_map}, paths["item_name_map_json"])
    _save_json(filter_log, paths["filter_log_json"])
    pd.DataFrame(factor_meta).to_csv(paths["factor_meta_csv"], index=False, encoding="utf-8-sig")

    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "response_long_df": response_long,
        "response_matrix_df": final_matrix,
        "item_meta_df": item_meta,
        "factor_meta_df": pd.DataFrame(factor_meta),
        "filter_log": filter_log,
        "summary": {
            "valid_qids": len(valid_qids),
            "response_rows": int(len(response_long)),
            "students": int(final_matrix.shape[0]),
            "items": int(final_matrix.shape[1]),
            "factors": int(len(factor_meta)),
        },
    }


# -------------------------
# Reporting helpers
# -------------------------
def load_factor_outputs(results_dir: str | os.PathLike) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    results_dir = Path(results_dir)
    scores_path = results_dir / "student_factor_scores.csv"
    item_path = results_dir / "item_params.csv"
    summary_path = results_dir / "model_summary.json"
    scores = pd.read_csv(scores_path, dtype={"student_id": str}) if scores_path.exists() else pd.DataFrame()
    item_params = pd.read_csv(item_path) if item_path.exists() else pd.DataFrame()
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return scores, item_params, summary


def build_student_knowledge_report(
    student_id: str,
    students_enriched: List[Dict[str, Any]],
    factor_scores_df: pd.DataFrame,
    factor_meta_df: pd.DataFrame,
    question_mid_map_payload: Dict[str, Any],
) -> Dict[str, Any]:
    sid = str(student_id)
    stu = next((x for x in students_enriched if str(x.get("学生id")) == sid), None)
    if stu is None:
        raise KeyError(f"未找到学生: {sid}")

    factor_row = pd.Series(dtype=object)
    if not factor_scores_df.empty:
        match = factor_scores_df[factor_scores_df["student_id"].astype(str) == sid]
        if not match.empty:
            factor_row = match.iloc[0]

    factor_meta_df = factor_meta_df.copy() if factor_meta_df is not None else pd.DataFrame()
    if not factor_meta_df.empty and not factor_scores_df.empty:
        score_cols = [c for c in factor_scores_df.columns if c.startswith("F")]
        population_mean = factor_scores_df[score_cols].mean(numeric_only=True)
        population_std = factor_scores_df[score_cols].std(numeric_only=True).replace(0, np.nan)
        rows: List[Dict[str, Any]] = []
        for _, meta in factor_meta_df.iterrows():
            fcol = str(meta["factor_name"])
            score = float(factor_row.get(fcol)) if fcol in factor_row.index and pd.notna(factor_row.get(fcol)) else np.nan
            z = (score - population_mean.get(fcol, np.nan)) / population_std.get(fcol, np.nan) if pd.notna(score) else np.nan
            mastery = float(np.clip(50 + 15 * z, 0, 100)) if pd.notna(z) else np.nan
            rows.append({
                "mid_id": str(meta["mid_id"]),
                "mid_name": str(meta["mid_name"]),
                "factor_name": fcol,
                "score": score,
                "zscore": z,
                "mastery": mastery,
                "n_items": int(meta.get("n_items", 0)),
            })
        mastery_df = pd.DataFrame(rows)
    else:
        mastery_df = pd.DataFrame(columns=["mid_id", "mid_name", "factor_name", "score", "zscore", "mastery", "n_items"])

    question_mid_map = question_mid_map_payload.get("question_mid_map", {})
    detail_rows: List[Dict[str, Any]] = []
    mid_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "correct": 0, "score_rates": []})
    for rec in stu.get("详细记录", []) or []:
        qid = str(rec.get("题目id_标准", rec.get("题目id")))
        mids = question_mid_map.get(qid, [])
        score_rate = float(rec.get("得分率", 0) or 0)
        is_correct = int(rec.get("是否正确", 0) or 0)
        for mid in mids:
            mid_stats[mid]["n"] += 1
            mid_stats[mid]["correct"] += is_correct
            mid_stats[mid]["score_rates"].append(score_rate)
            detail_rows.append({
                "student_id": sid,
                "qid": qid,
                "mid_id": mid,
                "score_rate": score_rate,
                "is_correct": is_correct,
            })

    detail_df = pd.DataFrame(detail_rows)
    if not mastery_df.empty:
        mastery_df["exercise_n"] = mastery_df["mid_id"].map(lambda x: mid_stats.get(x, {}).get("n", 0))
        mastery_df["exercise_acc"] = mastery_df["mid_id"].map(
            lambda x: mid_stats.get(x, {}).get("correct", 0) / mid_stats.get(x, {}).get("n", 1) if mid_stats.get(x, {}).get("n", 0) else np.nan
        )
        mastery_df["exercise_score_rate"] = mastery_df["mid_id"].map(
            lambda x: float(np.mean(mid_stats.get(x, {}).get("score_rates", []))) if mid_stats.get(x, {}).get("score_rates") else np.nan
        )
        mastery_df["weakness_index"] = (
            mastery_df["mastery"].rank(pct=True, ascending=True) * 0.6
            + mastery_df["exercise_acc"].fillna(mastery_df["exercise_acc"].mean()).rank(pct=True, ascending=True) * 0.4
        )
        mastery_df = mastery_df.sort_values(["mastery", "exercise_acc", "exercise_n"], ascending=[True, True, False])

    weak_df = mastery_df.head(8).copy() if not mastery_df.empty else mastery_df.copy()
    strong_df = mastery_df.sort_values(["mastery", "exercise_acc"], ascending=[False, False]).head(8).copy() if not mastery_df.empty else mastery_df.copy()

    return {
        "student": stu,
        "mastery_df": mastery_df,
        "weak_df": weak_df,
        "strong_df": strong_df,
        "detail_df": detail_df,
    }


def generate_pipeline(
    question_bank_source: str | os.PathLike | Any,
    students_source: str | os.PathLike | Any,
    taxonomy_source: str | os.PathLike | Any,
    work_dir: str | os.PathLike,
    irt_config: Optional[IRTConfig] = None,
    mirt_prep_config: Optional[MIRTPrepConfig] = None,
    rscript_executable: str = "Rscript",
    r_script_path: Optional[str | os.PathLike] = None,
) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
    work_dir = Path(work_dir)
    inputs_dir = work_dir / "inputs"
    mirt_input_dir = work_dir / "mid_mirt_input"
    mirt_results_dir = work_dir / "mirt_results"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    mirt_input_dir.mkdir(parents=True, exist_ok=True)
    mirt_results_dir.mkdir(parents=True, exist_ok=True)

    q_path = _copy_or_write_uploaded(question_bank_source, inputs_dir / "question_bank.json")
    s_path = _copy_or_write_uploaded(students_source, inputs_dir / "students_aggregated.json")
    x_path = _copy_or_write_uploaded(taxonomy_source, inputs_dir / "taxonomy.xlsx")

    yield {"stage": "load", "progress": 0.05, "message": "已接收 3 个输入文件，开始读取 JSON / Excel。"}
    question_bank_raw = _load_json_auto(q_path)
    students_raw = _load_json_auto(s_path)
    yield {"stage": "irt", "progress": 0.15, "message": "开始执行 IRT 预处理与 2PL 拟合。"}
    irt_result = run_pipeline(students_raw, question_bank_raw, config=irt_config or IRTConfig())

    enriched_students_path = work_dir / "students_enriched.json"
    enriched_items_path = work_dir / "question_bank_enriched.json"
    _save_json(irt_result["students_enriched"], enriched_students_path)
    _save_json(irt_result["question_bank_enriched"], enriched_items_path)
    yield {"stage": "irt", "progress": 0.35, "message": "IRT 完成，开始构造二级知识点 MIRT 输入矩阵。", "irt_metrics": irt_result["metrics"]}

    prep_result = prepare_mirt_inputs(students_raw, question_bank_raw, x_path, mirt_input_dir, cfg=mirt_prep_config or MIRTPrepConfig())
    yield {"stage": "mirt_prepare", "progress": 0.5, "message": "MIRT 输入文件已生成，开始调用 R/mirt 拟合。", "mirt_input_summary": prep_result["summary"]}

    if r_script_path is None:
        r_script_path = str(Path(__file__).with_name("run_mirt_integrated.R"))

    cmd = [
        rscript_executable,
        str(r_script_path),
        str(mirt_input_dir / "response_matrix.csv"),
        str(mirt_input_dir / "item_meta.csv"),
        str(mirt_input_dir / "mirt_model.txt"),
        str(mirt_input_dir / "question_mid_map.json"),
        str(mirt_results_dir),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    console_lines: List[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip("\n")
            console_lines.append(line)
            pct = 0.55 + min(0.35, len(console_lines) * 0.01)
            yield {"stage": "mirt_fit", "progress": pct, "message": line, "console": "\n".join(console_lines[-200:])}
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError("R/mirt 拟合失败。\n" + "\n".join(console_lines[-80:]))

    factor_scores_df, mirt_item_params_df, model_summary = load_factor_outputs(mirt_results_dir)
    factor_meta_payload = json.loads((mirt_input_dir / "question_mid_map.json").read_text(encoding="utf-8"))
    factor_meta_df = pd.DataFrame(factor_meta_payload.get("factor_meta", []))
    yield {"stage": "report", "progress": 0.95, "message": "MIRT 拟合完成，正在整理学生学情分析报告。"}

    result = {
        "work_dir": str(work_dir),
        "input_paths": {"question_bank": str(q_path), "students": str(s_path), "taxonomy": str(x_path)},
        "irt": irt_result,
        "mirt_prepare": prep_result,
        "mirt": {
            "student_factor_scores_df": factor_scores_df,
            "item_params_df": mirt_item_params_df,
            "model_summary": model_summary,
            "factor_meta_df": factor_meta_df,
            "console_lines": console_lines,
            "results_dir": str(mirt_results_dir),
        },
        "artifacts": {
            "students_enriched_json": str(enriched_students_path),
            "question_bank_enriched_json": str(enriched_items_path),
        },
    }
    yield {"stage": "done", "progress": 1.0, "message": "全部流程已完成。"}
    return result


def create_work_dir(base_dir: Optional[str | os.PathLike] = None) -> str:
    if base_dir:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("run_%Y%m%d_%H%M%S")
        path = Path(base_dir) / ts
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    return tempfile.mkdtemp(prefix="irt_mirt_run_")
