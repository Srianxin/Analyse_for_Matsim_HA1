# -*- coding: utf-8 -*-
"""
Task 3 – Multi-case analysis for directly affected agents
Cases: Base vs Policy-101 vs Policy-251

What this script does:
1) From events, identify "directly affected agents" (baseline- or per-case-based).
2) Filter output_trips.csv.gz for those agents.
3) Aggregate KPIs and export CSVs for your report.

NOTE: TRIPS MUST BE '...output_trips.csv.gz' (NOT output_links).
"""

import gzip
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Set

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

# ------------------------- CONFIG -------------------------

# ① 三个 case 的输入文件路径（改成你当前 Berlin 场景的输出路径）
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

# ② 定义“受影响”的目标 links（本作业：Ringbahnbrücke / Funkturm 周边 + 匝道）
TARGET_LINKS: Set[str] = {
    "428725175", "428725174", "305608389", "459811452", "4434516",
    "459811453#0", "4434508#0", "459810102", "459810103", "322543975",
    "227985279", "26144116", "169762615", "253770355", "4490226",
    "322543973", "84792456#0", "4490229", "4392640",
}

# ③ 判定口径：
# - "baseline"：用 Baseline 的 events 判定谁在被封闭走廊上出现过（推荐）
# - "per_case"：各 case 用自己的 events 判定（像“绕行走廊使用者”口径）
AFFECTED_DEFINITION = "baseline"  # "baseline" or "per_case"

# ④ 导出
WRITE_CSV = True
OUT_DIR = Path("./task3_outputs")  # 输出目录

# ------------------------- CORE -------------------------


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


# ---------- Robust trips reader (fixes your 'person' KeyError) ----------

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
            if df.shape[1] >= 5:  # 粗略判断读取是否合理
                break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f"Failed to read trips file: {trips_path}\nLast error: {last_err}")

    df.columns = [re.sub(r"\s+", " ", c).strip().lower() for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """从候选列表里选第一个存在的列名"""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of the candidate columns found. candidates={candidates}; "
        f"available={list(df.columns)[:30]}"
    )


def load_trips(trips_path: Path, affected: Set[str]) -> pd.DataFrame:
    """
    读取 trips，只保留受影响者；兼容不同版本列名。
    需要的逻辑列：person, trav_time, traveled_distance
    """
    df = _read_trips_flex(trips_path)

    person_col = _pick_col(df, ["person", "personid", "person_id", "agent", "agent_id"])
    trav_time_col = _pick_col(df, ["trav_time", "travel_time", "duration"])
    dist_col = _pick_col(df, ["traveled_distance", "distance", "leg_distance"])

    # 过滤受影响者
    df = df[df[person_col].astype(str).isin(affected)].copy()

    # 派生标准列
    df["trav_time_s"] = pd.to_timedelta(df[trav_time_col]).dt.total_seconds()
    df["dist_km"] = df[dist_col] / 1000.0

    # 统一列名，便于后续统计
    df = df.rename(columns={person_col: "person"})
    return df


def summarize(df: pd.DataFrame) -> pd.Series:
    """
    计算三项 KPI：
      - 直接受影响的唯一人数
      - 受影响人群总里程（km）
      - 受影响人群总出行时间（h）
    """
    return pd.Series(
        {
            "Directly affected agents": df["person"].nunique(),
            "km travelled by affected agents": df["dist_km"].sum(),
            "total time spent by affected agents (h)": df["trav_time_s"].sum() / 3600.0,
        }
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 判定受影响人群
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

    if WRITE_CSV:
        for name, ids in affected_by_case.items():
            pd.Series(sorted(ids), name="person").to_csv(OUT_DIR / f"affected_ids_{name}.csv", index=False)

    # 2) 读取 trips 并汇总
    summaries = {}
    detail_frames = []

    for name, paths in CASES.items():
        ids = affected_by_case[name]
        trips_df = load_trips(paths["trips"], ids)
        summaries[name] = summarize(trips_df)
        trips_df = trips_df.assign(run=name)
        detail_frames.append(trips_df)

    summary_tbl = pd.DataFrame(summaries).T

    # 3) 计算相对 Base 的百分比变化
    if "Base" in summary_tbl.index:
        base_row = summary_tbl.loc["Base"]
        diff_tbl = (summary_tbl.subtract(base_row, axis=1)).divide(base_row, axis=1) * 100.0
        diff_tbl = diff_tbl.rename(index=lambda x: f"{x} vs Base (%)" if x != "Base" else x)
        display_tbl = pd.concat([summary_tbl.loc[["Base"]], diff_tbl.loc[[i for i in diff_tbl.index if i != "Base"]]])
    else:
        display_tbl = summary_tbl.copy()

    # 4) 打印 + 导出
    def _fmt(v):
        if isinstance(v, (int,)):
            return f"{v:,d}"
        try:
            return f"{float(v):,.1f}"
        except Exception:
            return str(v)

    print("\n=== Directly affected agents: Summary ===\n")
    print(display_tbl.applymap(_fmt))

    if WRITE_CSV:
        summary_tbl.to_csv(OUT_DIR / "affected_summary_raw.csv")
        display_tbl.to_csv(OUT_DIR / "affected_summary_for_report.csv", float_format="%.4f")
        pd.concat(detail_frames, ignore_index=True).to_csv(OUT_DIR / "affected_agents_detail.csv", index=False)


if __name__ == "__main__":
    main()

    # === Plot 1: % change vs Base (bar plot) ===
    import matplotlib.pyplot as plt

    # 如果你是从 CSV 读：
    # import pandas as pd
    # display_tbl = pd.read_csv("task3_outputs/affected_summary_for_report.csv", index_col=0)

    pct = display_tbl.loc[display_tbl.index.str.contains("vs Base"),
    ["km travelled by affected agents", "total time spent by affected agents (h)"]]

    plt.figure()
    pct.plot(kind="bar")
    plt.title("Policy vs Base: % change for directly affected agents")
    plt.ylabel("% change")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # === Plot 2: Baseline absolute values (agents, km, h) ===
    base_vals = display_tbl.loc["Base",
    ["Directly affected agents", "km travelled by affected agents", "total time spent by affected agents (h)"]]

    plt.figure()
    base_vals.plot(kind="bar")
    plt.title("Baseline – directly affected agents (absolute values)")
    plt.ylabel("Value")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()