# -*- coding: utf-8 -*-
"""
Task 3
Cases: Base vs Policy-101 vs Policy-251


Author: Group 8
"""

import gzip
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

# ------------------------- CONFIG -------------------------

# 三个 case 的输入文件路径（保持你之前的路径）
CASES = {
    "Base": {
        "events": Path("/Users/srainxin/Downloads/berlin-v6.4-10pct/berlin-v6.4.output_events.xml.gz"),
        "trips":  Path("/Users/srainxin/Downloads/berlin-v6.4-10pct/berlin-v6.4.output_trips.csv.gz"),
    },
    "Policy-101": {
        "events": Path("/Users/srainxin/Downloads/berlin-v6.4-10pct-101/berlin-v6.4.output_events.xml.gz"),
        "trips":  Path("/Users/srainxin/Downloads/berlin-v6.4-10pct-101/berlin-v6.4.output_trips.csv.gz"),
    },
    "Policy-251": {
        "events": Path("/Users/srainxin/Downloads/berlin-v6.4-10pct-251/berlin-v6.4.output_events.xml.gz"),
        "trips":  Path("/Users/srainxin/Downloads/berlin-v6.4-10pct-251/berlin-v6.4.output_trips.csv.gz"),
    },
}

# Ringbahnbrücke / Funkturm 周边 links（用于“受影响者”识别）
TARGET_LINKS: Set[str] = {
    "428725175", "428725174", "305608389", "459811452", "4434516",
    "459811453#0", "4434508#0", "459810102", "459810103", "322543975",
    "227985279", "26144116", "169762615", "253770355", "4490226",
    "322543973", "84792456#0", "4490229", "4392640",
}

# 受影响人群判定口径（推荐 baseline）
AFFECTED_DEFINITION = "baseline"   # "baseline" or "per_case"

# 峰时段窗口（小时）
AM_START, AM_END = 7, 10
PM_START, PM_END = 16, 19

# 导出
WRITE_CSV = True
OUT_DIR = Path("./task3_outputs")  # 输出目录

# ------------------------- HELPERS -------------------------


def _assert_py312():
    # 保险：限定 3.12（可注释）
    if sys.version_info[:2] != (3, 12):
        print(f"[WARN] Python {sys.version.split()[0]} detected. Recommended 3.12.x.", file=sys.stderr)


def parse_affected_from_events(events_path: Path, target_links: Set[str]) -> Set[str]:
    """
    在给定 events 中，找出进入过 target_links 的所有 person（或由 vehicle 映射到 person）。
    兼容 'linkEnter' 与 'entered link' 两种事件名。
    """
    veh2driver: Dict[str, str] = {}
    affected: Set[str] = set()

    opener = gzip.open if events_path.suffix == ".gz" else open
    with opener(events_path, "rt", encoding="utf-8") as fh:
        for _, ev in ET.iterparse(fh, events=("end",)):
            if ev.tag != "event":
                continue
            typ = ev.get("type")

            if typ == "PersonEntersVehicle":
                veh = ev.get("vehicle")
                per = ev.get("person")
                if veh and per:
                    veh2driver[veh] = per

            elif typ in {"linkEnter", "entered link"}:
                link_id = ev.get("link")
                if link_id in target_links:
                    per = ev.get("person")
                    if not per:
                        veh = ev.get("vehicle")
                        per = veh2driver.get(veh) if veh else None
                    if per:
                        affected.add(str(per))

            ev.clear()

    return affected


# ---------- Robust trips reader (兼容不同列名/分隔符/单位) ----------

def _read_trips_flex(trips_path: Path) -> pd.DataFrame:
    """
    更健壮地读取 MATSim trips：
    - 自动尝试分隔符（; , 或 \t）
    - 列名统一为小写、去空格
    """
    last_err = None
    for sep_try in [None, ";", ",", "\t"]:
        try:
            df = pd.read_csv(trips_path, sep=sep_try, compression="gzip", engine="python")
            if df.shape[1] >= 5:
                break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f"Failed to read trips file: {trips_path}\nLast error: {last_err}")

    df.columns = [re.sub(r"\s+", " ", c).strip().lower() for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing columns {candidates}; have {list(df.columns)[:25]}")


def _to_seconds(series: pd.Series) -> pd.Series:
    """
    将可能是数字秒、'PTxxMxxS'、'HH:MM:SS' 的列转成秒(float)
    """
    # 若全是数字或可转数字，直接视为秒
    try:
        as_num = pd.to_numeric(series, errors="raise")
        return as_num.astype(float)
    except Exception:
        pass
    # 否则当作时间字符串交给 pandas
    return pd.to_timedelta(series).dt.total_seconds()


def load_trips_generic(trips_path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    读取 trips 并标准化关键列名，返回 DataFrame + 使用的列名映射
    产出标准列：
      - 'person'（字符串）
      - 'dep_s'（出发秒）
      - 'trav_time_s'（出行时间秒）
      - 'dist_km'（距离 km）
    """
    df = _read_trips_flex(trips_path)
    person_col = _pick_col(df, ["person", "personid", "person_id", "agent", "agent_id"])
    dep_col = _pick_col(df, ["dep_time", "departure_time", "start_time"])
    trav_col = _pick_col(df, ["trav_time", "travel_time", "duration"])
    dist_col = _pick_col(df, ["traveled_distance", "distance", "leg_distance"])

    out = pd.DataFrame()
    out["person"] = df[person_col].astype(str)
    out["dep_s"] = _to_seconds(df[dep_col])
    out["trav_time_s"] = _to_seconds(df[trav_col])
    out["dist_km"] = df[dist_col].astype(float) / 1000.0

    return out, {"person": person_col, "dep_time": dep_col, "trav_time": trav_col, "distance": dist_col}


def add_peak_flags(df: pd.DataFrame,
                   am: Tuple[int, int] = (AM_START, AM_END),
                   pm: Tuple[int, int] = (PM_START, PM_END)) -> pd.DataFrame:
    """
    以出发时刻判定峰时段
    """
    s = df.copy()
    def in_window(sec, start_h, end_h):
        return (sec >= start_h*3600) & (sec < end_h*3600)
    s["is_am_peak"] = in_window(s["dep_s"], am[0], am[1])
    s["is_pm_peak"] = in_window(s["dep_s"], pm[0], pm[1])
    s["is_offpeak"] = ~(s["is_am_peak"] | s["is_pm_peak"])
    return s


# ------------------------- KPI BUILDERS -------------------------

def summarize_directly_affected(df: pd.DataFrame) -> pd.Series:
    """
    受影响者三项 KPI（与你之前一致）
    """
    return pd.Series({
        "Directly affected agents": df["person"].nunique(),
        "km travelled by affected agents": df["dist_km"].sum(),
        "total time spent by affected agents (h)": df["trav_time_s"].sum()/3600.0,
    })


def kpi_stats_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成四行（AM/PM/Off/All）的 KPI：
      - trip_count
      - total_hours
      - mean_trav_min, median_trav_min, p95_trav_min
      - total_km
    """
    def stats(x: pd.DataFrame) -> dict:
        if len(x) == 0:
            return dict(trip_count=0, total_hours=0.0,
                        mean_trav_min=0.0, median_trav_min=0.0, p95_trav_min=0.0,
                        total_km=0.0)
        return dict(
            trip_count=len(x),
            total_hours=x["trav_time_s"].sum()/3600.0,
            mean_trav_min=x["trav_time_s"].mean()/60.0,
            median_trav_min=x["trav_time_s"].median()/60.0,
            p95_trav_min=float(np.percentile(x["trav_time_s"], 95))/60.0,
            total_km=x["dist_km"].sum(),
        )

    rows = {
        "AM peak (07–10)": stats(df[df["is_am_peak"]]),
        "PM peak (16–19)": stats(df[df["is_pm_peak"]]),
        "Off-peak":        stats(df[df["is_offpeak"]]),
        "All day":         stats(df),
    }
    return pd.DataFrame(rows).T


def percent_change_vs_base(tbl: pd.DataFrame, base_name: str = "Base") -> pd.DataFrame:
    """
    将同结构的 KPI 表（index=case，columns=指标）转换成：
      Base 行 = 绝对值；其他行 = (Policy-Base)/Base * 100
    """
    if base_name not in tbl.index:
        return tbl.copy()
    base = tbl.loc[base_name]
    diff = (tbl.subtract(base, axis=1)).divide(base.replace(0, np.nan), axis=1) * 100.0
    out = pd.concat([tbl.loc[[base_name]], diff.drop(index=base_name)])
    out = out.rename(index=lambda i: f"{i} vs Base (%)" if i != base_name else i)
    return out


def _format_df_for_print(df: pd.DataFrame) -> pd.DataFrame:
    def _fmt(v):
        try:
            fv = float(v)
            # 整数显示为无小数，其他一位小数
            if abs(fv - round(fv)) < 1e-9:
                return f"{int(round(fv)):,}"
            return f"{fv:,.1f}"
        except Exception:
            return str(v)
    # pandas 未来弃用 applymap；这里用按列 map
    return df.apply(lambda col: col.map(_fmt))


# ------------------------- MAIN -------------------------

def main():
    _assert_py312()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 受影响者 ID 集合
    if AFFECTED_DEFINITION == "baseline":
        affected = parse_affected_from_events(CASES["Base"]["events"], TARGET_LINKS)
        affected_by_case = {name: affected for name in CASES.keys()}
        print(f"[INFO] Affected set (baseline-based): {len(affected):,} agents")
    elif AFFECTED_DEFINITION == "per_case":
        affected_by_case = {
            name: parse_affected_from_events(paths["events"], TARGET_LINKS)
            for name, paths in CASES.items()
        }
        for name, ids in affected_by_case.items():
            print(f"[INFO] Affected set ({name}): {len(ids):,} agents")
    else:
        raise ValueError("AFFECTED_DEFINITION must be 'baseline' or 'per_case'.")

    # 2) 读取 trips（全体 & 受影响者），加峰时段标记
    trips_all_by_case: dict[str, pd.DataFrame] = {}
    trips_aff_by_case: dict[str, pd.DataFrame] = {}

    for name, paths in CASES.items():
        trips_df, used_cols = load_trips_generic(paths["trips"])
        trips_df = add_peak_flags(trips_df)

        trips_all_by_case[name] = trips_df

        ids = affected_by_case[name]
        trips_aff_by_case[name] = trips_df[trips_df["person"].isin(ids)].copy()

    # 3) ——(A) 受影响者“总表”（与你之前一致）——
    aff_summary = {}
    for name, df in trips_aff_by_case.items():
        aff_summary[name] = summarize_directly_affected(df)
    aff_summary_tbl = pd.DataFrame(aff_summary).T
    aff_display = percent_change_vs_base(aff_summary_tbl, base_name="Base")

    print("\n=== Directly affected agents: Summary ===\n")
    print(_format_df_for_print(aff_display))

    if WRITE_CSV:
        aff_summary_tbl.to_csv(OUT_DIR / "affected_summary_raw.csv")
        aff_display.to_csv(OUT_DIR / "affected_summary_for_report.csv", float_format="%.6f")
        # 明细
        pd.concat([df.assign(run=name) for name, df in trips_aff_by_case.items()],
                  ignore_index=True).to_csv(OUT_DIR / "affected_agents_detail.csv", index=False)

    # 4) ——(B) 峰时段 KPI：全体出行者——（无 stack/unstack，长表法更稳）
    # 先得到每个 case 的 period KPI（行=Period）
    all_kpi_by_case = {name: kpi_stats_block(df).reset_index().rename(columns={"index": "Period"})
                       for name, df in trips_all_by_case.items()}

    base_all = all_kpi_by_case["Base"].copy()
    base_all["Scenario"] = "Base"

    rows_all = [base_all]  # 第一部分：Base 绝对值（便于报告复用）
    # 第二部分：Policy 相对 Base 的 % 变化
    for name, tbl in all_kpi_by_case.items():
        if name == "Base":
            continue
        merged = tbl.merge(base_all.drop(columns=["Scenario"]), on="Period", suffixes=("", "_base"))
        pc = {}
        for col in ["trip_count", "total_hours", "mean_trav_min", "median_trav_min", "p95_trav_min", "total_km"]:
            pc[col] = (merged[col] - merged[f"{col}_base"]) / merged[f"{col}_base"] * 100.0
        out = pd.DataFrame({"Period": merged["Period"], **pc})
        out["Scenario"] = f"{name} vs Base (%)"
        rows_all.append(out)

    all_vs_base_tbl = pd.concat(rows_all, ignore_index=True)
    all_vs_base_tbl = all_vs_base_tbl.set_index(["Scenario", "Period"])

    print("\n=== Peak-hour KPIs (ALL travellers): Policy vs Base (%) by period ===\n")
    print(_format_df_for_print(all_vs_base_tbl))

    if WRITE_CSV:
        # 原值（各 scenario 各 period 的绝对值）
        pd.concat(
            [df.assign(Scenario=name) for name, df in all_kpi_by_case.items()],
            ignore_index=True
        ).to_csv(OUT_DIR / "peak_all_raw_by_scenario_period.csv", index=False)

        # 含 Base 绝对值 + Policy 相对 Base（%）
        all_vs_base_tbl.to_csv(OUT_DIR / "peak_all_vs_base.csv", float_format="%.6f")

    # 5) ——(C) 峰时段 KPI：受影响者——（同样用长表法）
    aff_kpi_by_case = {name: kpi_stats_block(df).reset_index().rename(columns={"index": "Period"})
                       for name, df in trips_aff_by_case.items()}

    base_aff = aff_kpi_by_case["Base"].copy()
    base_aff["Scenario"] = "Base"

    rows_aff = [base_aff]
    for name, tbl in aff_kpi_by_case.items():
        if name == "Base":
            continue
        merged = tbl.merge(base_aff.drop(columns=["Scenario"]), on="Period", suffixes=("", "_base"))
        pc = {}
        for col in ["trip_count", "total_hours", "mean_trav_min", "median_trav_min", "p95_trav_min", "total_km"]:
            pc[col] = (merged[col] - merged[f"{col}_base"]) / merged[f"{col}_base"] * 100.0
        out = pd.DataFrame({"Period": merged["Period"], **pc})
        out["Scenario"] = f"{name} vs Base (%)"
        rows_aff.append(out)

    aff_vs_base_tbl = pd.concat(rows_aff, ignore_index=True)
    aff_vs_base_tbl = aff_vs_base_tbl.set_index(["Scenario", "Period"])

    print("\n=== Peak-hour KPIs (DIRECTLY AFFECTED): Policy vs Base (%) by period ===\n")
    print(_format_df_for_print(aff_vs_base_tbl))

    if WRITE_CSV:
        pd.concat(
            [df.assign(Scenario=name) for name, df in aff_kpi_by_case.items()],
            ignore_index=True
        ).to_csv(OUT_DIR / "peak_aff_raw_by_scenario_period.csv", index=False)

        aff_vs_base_tbl.to_csv(OUT_DIR / "peak_aff_vs_base.csv", float_format="%.6f")

    # 6) 附：导出每个 case 的受影响 ID（口径检查）
    if WRITE_CSV:
        for name, ids in affected_by_case.items():
            pd.Series(sorted(ids), name="person").to_csv(OUT_DIR / f"affected_ids_{name}.csv", index=False)


if __name__ == "__main__":
    main()