import xml.etree.ElementTree as ET
import gzip

POLICY_LINKS = {"myNewHighway1", "myNewHighway1Rev",
    "myNewHighway2", "myNewHighway3Rev",
    "myNewHighway4", "myNewHighway4Rev",
    "myNewHighway5", "myNewHighway5Rev",
    "myNewHighway6", "myNewHighway6Rev",
    "myNewHighway7", "myNewHighway7Rev",
    "myNewHighway8", "myNewHighway8Rev",
    "myNewHighway9", "myNewHighway9Rev",
    "myNewHighway10", "myNewHighway10Rev",
    "myNewHighway11", "myNewHighway11Rev",
    "myNewHighway12", "myNewHighway12Rev"}             # 高速 linkId 集合
EVENTS       = "/Users/srainxin/Downloads/output-kelheim-v3.1-1pct/kelheim-v3.1-1pct.output_events.xml.gz"

veh2driver = {}
affected   = set()               # PersonId 集合

open_fn = gzip.open if EVENTS.endswith(".gz") else open
with open_fn(EVENTS, "rt") as fh:
    for _, ev in ET.iterparse(fh):
        if ev.tag != "event":
            ev.clear(); continue

        typ = ev.get("type")

        # 1. 建立 vehicle → driver/person 的映射
        if typ == "PersonEntersVehicle":
            veh2driver[ev.get("vehicle")] = ev.get("person")

        # 2. LinkEnter: 只有 vehicle; 要通过映射找到 person
        elif typ == "entered link" and ev.get("link") in POLICY_LINKS:
            person = ev.get("person")            # walk/bike 情况
            if person is None:                   # 开车/乘车
                person = veh2driver.get(ev.get("vehicle"))
            if person:
                affected.add(person)

        ev.clear()

print(f"Affected agents: {len(affected)}")