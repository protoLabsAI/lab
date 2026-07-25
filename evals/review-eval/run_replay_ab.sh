#!/usr/bin/env bash
# Scored fast-vs-smart replay A/B over the pinned manifest (protoLab#26).
#
#   VERA_API_KEY=... ./run_replay_ab.sh
#
# Env: VERA_API_KEY (required — Infisical, Vera stack), VERA_HOST (default
# http://ava:7874), TRIALS (default 2), MODELS (default "protolabs/fast protolabs/smart"),
# OUT (default runs-<stamp>). Endpoint per qaEngineer#20: POST /api/plugins/pr-reviewer/replay
# {"manifest": [rows], "model": ..., "trials": N} -> {"runs": [...]}; include_raw needs
# pr-reviewer#43 (harmless to request before it's deployed).
set -euo pipefail
cd "$(dirname "$0")"
: "${VERA_API_KEY:?set VERA_API_KEY (Infisical: Vera stack A2A_AUTH_TOKEN)}"
HOST="${VERA_HOST:-http://ava:7874}"
TRIALS="${TRIALS:-2}"
MODELS="${MODELS:-protolabs/fast protolabs/smart}"
OUT="${OUT:-runs-$(date +%Y%m%d-%H%M)}"
mkdir -p "$OUT"

MANIFEST=$(python3 -c "import json; print(json.dumps([json.loads(l) for l in open('replay_manifest.jsonl') if l.strip()]))")

for model in $MODELS; do
  slug=${model##*/}
  echo "== $model × $TRIALS trials → $OUT/$slug.json =="
  curl -sS --fail-with-body -m 14400 -X POST "$HOST/api/plugins/pr-reviewer/replay" \
    -H "Authorization: Bearer $VERA_API_KEY" -H "X-API-Key: $VERA_API_KEY" \
    -H 'Content-Type: application/json' \
    -d "{\"manifest\": $MANIFEST, \"model\": \"$model\", \"trials\": $TRIALS, \"include_raw\": true}" \
    > "$OUT/$slug.json"
  python3 -c "import json; d=json.load(open('$OUT/$slug.json')); print('   runs returned:', len(d.get('runs', d if isinstance(d, list) else [])))"
done

# Split each {"runs": [...]} response into per-run files (the scorer's native shape).
python3 - "$OUT" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
for f in sorted(out.glob('*.json')):
    if f.name.startswith('run_'):
        continue
    d = json.loads(f.read_text())
    runs = d.get('runs', d if isinstance(d, list) else [])
    for i, run in enumerate(runs):
        (out / f"run_{f.stem}_{i:03d}.json").write_text(json.dumps(run))
    print(f"{f.name}: split {len(runs)} runs")
EOF

python3 score_ab.py --truth truth.jsonl "$OUT"/run_*.json | tee "$OUT/scorecard.txt"
echo; echo "scorecard: $OUT/scorecard.txt"
