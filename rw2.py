import gzip, xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

# Wrong affected agents! I cant solve this problem, but you can use link.py to get the correct number.

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
# ---------- 文件路径 ----------
BASE_EVENTS   = Path("/Users/srainxin/Downloads/output-kelheim-v3.1-1pct-base/kelheim-v3.1-1pct-base.output_events.xml.gz")
POLICY_EVENTS = Path("/Users/srainxin/Downloads/output-kelheim-v3.1-1pct/kelheim-v3.1-1pct.output_events.xml.gz")
BASE_TRIPS    = Path("/Users/srainxin/Downloads/output-kelheim-v3.1-1pct-base/kelheim-v3.1-1pct-base.output_trips.csv.gz")
POLICY_TRIPS  = Path("/Users/srainxin/Downloads/output-kelheim-v3.1-1pct/kelheim-v3.1-1pct.output_trips.csv.gz")

HIGHWAY_LINKS = {
    "myNewHighway1","myNewHighway1Rev","myNewHighway2","myNewHighway3Rev",
    "myNewHighway4","myNewHighway4Rev","myNewHighway5","myNewHighway5Rev",
    "myNewHighway6","myNewHighway6Rev","myNewHighway7","myNewHighway7Rev",
    "myNewHighway8","myNewHighway8Rev","myNewHighway9","myNewHighway9Rev",
    "myNewHighway10","myNewHighway10Rev","myNewHighway11","myNewHighway11Rev",
    "myNewHighway12","myNewHighway12Rev"
}
# ---------- 抽取受影响名单，只扫 policy-events ----------
def get_affected(events_path: Path) -> set[str]:
    veh2driver, affected = {}, set()
    opener = gzip.open if events_path.suffix == ".gz" else open
    with opener(events_path, "rt") as fh:
        # 只监听 end 事件，防抖
        for _, ev in ET.iterparse(fh, events=("end",)):
            if ev.tag != "event": continue
            typ = ev.get("type")

            if typ == "PersonEntersVehicle":
                veh2driver[ev.get("vehicle")] = ev.get("person")

            elif typ in {"entered link", "linkEnter"} and ev.get("link") in HIGHWAY_LINKS:
                pid = ev.get("person") or veh2driver.get(ev.get("vehicle"))
                if pid: affected.add(pid)

            ev.clear()
    return affected

affected = get_affected(POLICY_EVENTS)
print(f"受影响 agent 数量 (policy run)：{len(affected):,}")

# ---------- 读取 trips & 过滤 ----------
def trips_for(path: Path, affected: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", compression="gzip", engine="python")
    df = df[df["person"].astype(str).isin(affected)].copy()

    df["trav_time_s"] = pd.to_timedelta(df["trav_time"]).dt.total_seconds()
    df["dist_km"]     = df["traveled_distance"] / 1000.0
    return df

base_df   = trips_for(BASE_TRIPS,   affected)
pol_df    = trips_for(POLICY_TRIPS, affected)

# ---------- 汇总 ----------
def summary(df: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "Directly affected agents":            df["person"].nunique(),
        "km travelled by aff. agents":         df["dist_km"].sum(),
        "total time spent by aff. Agents (h)": df["trav_time_s"].sum()/3600.0,
    })

tbl = pd.concat(
    {"Base Case": summary(base_df), "Policy Case": summary(pol_df)},
    axis=1
).T

tbl.loc["Difference (%)"] = (tbl.loc["Policy Case"] - tbl.loc["Base Case"]) / tbl.loc["Base Case"] * 100

# ---------- 漂亮打印 ----------
tbl_fmt = tbl.applymap(lambda v: f"{v:,.1f}" if isinstance(v, float) else f"{int(v):,}")
print("\nComparison table\n", tbl_fmt)