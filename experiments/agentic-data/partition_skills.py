"""Partition Ornith τ-bench retail trajectories into skill clusters (arm-E)."""
import json, glob
SKILL = {  # primary action -> skill cluster
  "cancel_pending_order":"cancel", "return_delivered_order_items":"return",
  "exchange_delivered_order_items":"exchange",
  "modify_pending_order_items":"modify","modify_pending_order_address":"modify",
  "modify_pending_order_payment":"modify",
}
def canon(entry, skill):
    traj=entry.get("traj") or []
    msgs=[]
    for m in traj:
        if m.get("role") not in ("system","user","assistant","tool"): continue
        tcs=None
        if m.get("tool_calls"):
            tcs=[{"id":t.get("id"),"name":t["function"]["name"],"arguments":t["function"]["arguments"]} for t in m["tool_calls"]]
        msgs.append({"role":m["role"],"content":m.get("content"),"tool_calls":tcs,"tool_call_id":m.get("tool_call_id")})
    while msgs and msgs[-1]["role"]!="assistant": msgs.pop()
    if len(msgs)<2: return None
    return {"id":f"ornith_tau__{skill}_{entry.get('task_id')}","source":"ornith_tau","teacher":"ornith-35b",
            "domain":f"retail_{skill}","messages":msgs,"tools":[],"verified":True,"reward":1.0,"split":"train"}
buckets={}
for f in glob.glob("/mnt/data/datasets/agentic-distill/_raw/tau-retail/*.json"):
    for e in json.load(open(f)):
        if float(e.get("reward",0))<1.0: continue
        acts=(e.get("info",{}) or {}).get("task",{}).get("actions") or e.get("task",{}).get("actions") or []
        names=[a.get("name") for a in acts if isinstance(a,dict) and a.get("name")]
        sk=SKILL.get(names[0]) if names else None
        if not sk: continue
        row=canon(e,sk)
        if row: buckets.setdefault(sk,[]).append(row)
import os; os.makedirs("/mnt/data/datasets/agentic-distill/_raw/skills",exist_ok=True)
for sk,rows in buckets.items():
    with open(f"/mnt/data/datasets/agentic-distill/_raw/skills/{sk}.jsonl","w") as fo:
        for r in rows: fo.write(json.dumps(r)+"\n")
    print(f"  {sk}: {len(rows)}")
