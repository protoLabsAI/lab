#!/usr/bin/env python3
"""Score generated completions against human references with the Rosmine metrics.

Per model vs human:  MMD (embed), Token-L2 (word dist), slop rate, self-BLEU, JMQ.
If two models given: direct pairwise win rate (DG vs Gemma4) too.
JMQ = 2 x win-rate-vs-human (1.0 = indistinguishable from human; >1 = preferred).
"""
import argparse, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import chat, embed
import metrics as M

HERE = os.path.dirname(__file__)
JUDGE_EP = "http://localhost:8000/v1/chat/completions"
JUDGE_MODEL = "local"

JUDGE = ('Given a writing prompt and two responses, choose which is the better piece of '
         'writing (quality of prose AND following the prompt). Reply with ONLY a JSON '
         'object: {{"winner":"A"}} or {{"winner":"B"}}.\n\n'
         'PROMPT:\n{p}\n\nRESPONSE A:\n{a}\n\nRESPONSE B:\n{b}')

def judge(prompt, a, b):
    """Return 1 if A wins, 0 if B wins (order randomized internally)."""
    sw = random.random() < 0.5
    x, y = (b, a) if sw else (a, b)
    r, _, _ = chat(JUDGE_EP, JUDGE_MODEL, JUDGE.format(p=prompt[:2000], a=x[:4000], b=y[:4000]),
                   max_tokens=80, temperature=0.0, no_think=True)
    m = re.search(r'"winner"\s*:\s*"?([AB])"?', r, re.I)
    w = m.group(1).upper() if m else random.choice("AB")
    a_won = (w == "B") if sw else (w == "A")
    return 1 if a_won else 0

def winrate(prompts, A, B):
    """Fraction of prompts where A beats B."""
    return sum(judge(p, a, b) for p, a, b in zip(prompts, A, B)) / max(len(prompts), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default=os.path.join(HERE, "data", "human_refs.jsonl"))
    ap.add_argument("--gen", nargs="+", required=True, help="gen_<label>.jsonl files")
    a = ap.parse_args()

    refs = {r["id"]: r for r in (json.loads(l) for l in open(a.refs))}
    models = {}
    for g in a.gen:
        label = re.sub(r"^gen_|\.jsonl$", "", os.path.basename(g))
        rows = [json.loads(l) for l in open(g)]
        models[label] = {r["id"]: r["output"] for r in rows}

    # common id set across human + all models
    ids = sorted(set(refs) & set.intersection(*[set(m) for m in models.values()]))
    human = [refs[i]["human"] for i in ids]
    prompts = [refs[i]["prompt"] for i in ids]
    print(f"scoring {len(ids)} prompts across: {list(models)} + human\n")

    hum_emb = embed(human)
    report = {"n": len(ids), "models": {}}
    for label, m in models.items():
        outs = [m[i] for i in ids]
        emb = embed(outs)
        print(f"[{label}] judging vs human ({len(ids)} pairwise)...", flush=True)
        wr = winrate(prompts, outs, human)
        report["models"][label] = {
            "MMD_vs_human": round(M.mmd2(emb, hum_emb), 4),
            "TokenL2_vs_human": round(M.token_l2(outs, human), 4),
            "JMQ": round(2 * wr, 3), "winrate_vs_human": round(wr, 3),
            "slop": M.slop_rate(outs), "self_bleu": M.self_bleu(outs),
            "mean_tok_s": round(sum(json.loads(l).get("tok_s", 0)
                                    for l in open([g for g in a.gen if label in g][0])) / len(ids), 1),
        }

    report["human_ref"] = {"slop": M.slop_rate(human), "self_bleu": M.self_bleu(human),
                           "TokenL2_self": 0.0}
    if len(models) == 2:
        (la, ma), (lb, mb) = list(models.items())
        wr = winrate(prompts, [ma[i] for i in ids], [mb[i] for i in ids])
        report["head_to_head"] = {f"{la}_vs_{lb}_winrate": round(wr, 3)}

    out = os.path.join(HERE, "out", "scorecard.json")
    json.dump(report, open(out, "w"), indent=2)

    print("\n" + "=" * 74)
    print(f"{'model':<10}{'MMD↓':>9}{'TokenL2↓':>10}{'JMQ↑':>7}{'slop/1k↓':>10}{'selfBLEU':>10}{'tok/s':>8}")
    print("-" * 74)
    for label, r in report["models"].items():
        print(f"{label:<10}{r['MMD_vs_human']:>9}{r['TokenL2_vs_human']:>10}{r['JMQ']:>7}"
              f"{r['slop']['total_per_1k']:>10}{r['self_bleu']:>10}{r['mean_tok_s']:>8}")
    h = report["human_ref"]
    print(f"{'human':<10}{'0.0':>9}{'0.0':>10}{'1.0':>7}{h['slop']['total_per_1k']:>10}{h['self_bleu']:>10}{'-':>8}")
    if "head_to_head" in report:
        print("-" * 74); print("head-to-head:", report["head_to_head"])
    print("=" * 74)
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()
