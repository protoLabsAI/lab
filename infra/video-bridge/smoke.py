"""End-to-end smoke: POST a job, poll to completion, fetch mp4 bytes."""
import time, sys, httpx
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100"
c = httpx.Client(timeout=30.0)

r = c.post(f"{BASE}/v1/videos", json={
    "model": "protolabs/ltx2-distilled",
    "prompt": "a red fox trotting through fresh snow at dawn, breath visible, cinematic, soft light",
    "seconds": 2, "size": "768x512", "seed": 123,
})
print("POST", r.status_code, r.json())
r.raise_for_status()
jid = r.json()["id"]

t0 = time.time()
while True:
    time.sleep(2)
    s = c.get(f"{BASE}/v1/videos/{jid}").json()
    print(f"  [{time.time()-t0:4.0f}s] status={s['status']} progress={s.get('progress')}")
    if s["status"] in ("completed", "failed"):
        break
    if time.time() - t0 > 300:
        print("TIMEOUT"); sys.exit(1)

if s["status"] == "failed":
    print("FAILED:", s.get("error")); sys.exit(1)

r = c.get(f"{BASE}/v1/videos/{jid}/content")
print("GET /content:", r.status_code, r.headers.get("content-type"), len(r.content), "bytes")
open("/mnt/data/ltx-out/bridge_smoke.mp4", "wb").write(r.content)
print("saved /mnt/data/ltx-out/bridge_smoke.mp4")
