"""Self-MoA (fusion) over the reasoning lane on reasoning_hard — bench vs solo.

Replicates protolabs/fusion-reasoning locally against the live reasoning lane
(ThinkingCap :8041): N candidates @ T=0.7 (thinking), then a T=0 judge synthesizes
the final answer. Graded with the same MatchGrader as run_custom, so it compares
directly to the solo ThinkingCap reasoning_hard run (0.880).

Usage:
  python -m runners.fusion_reasoning --base-url http://127.0.0.1:8041/v1 --model reasoning \
      --panel 3 --output-dir results/fusion-reasoning
"""
from __future__ import annotations
import glob, json, os, time, click
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
from openai import OpenAI
from graders.match import MatchGrader

TASKS_DIR = "/home/ava/dev/lab/evals/tasks/reasoning_hard"

AGG_TMPL = """You are given a reasoning problem and {n} independent candidate solutions. Some may be wrong. Work out the correct answer, using the candidates as hints but verifying each step yourself. End with the final answer in EXACTLY the format the problem requests.

# Problem
{problem}

# Candidate solutions
{cands}

Now give the correct, fully-worked final answer."""

def gen(client, model, prompt, temp, max_tokens):
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=temp, top_p=0.95, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20, "min_p": 0.0},
        )
        m = r.choices[0].message
        c = m.content or ""
        rc = getattr(m, "reasoning_content", None) or ""
        return (c + ("\n" + rc if not c else "")) or rc  # answer usually in content; fall back to reasoning
    except Exception as e:
        return f"__ERR__ {e}"

def grade(task, output):
    scores = []
    for gc in task.get("graders", []):
        if gc["type"] != "match":
            continue
        g = MatchGrader(dimension=gc.get("dimension", "match"), mode=gc.get("mode", "contains"),
                        expected=gc.get("expected"), case_sensitive=gc.get("case_sensitive", True),
                        tolerance=gc.get("tolerance", 0.0))
        scores.append(g.grade({"prompt": task["prompt"]}, {"output": output}).score)
    return sum(scores) / len(scores) if scores else 0.0

@click.command()
@click.option("--base-url", default="http://127.0.0.1:8041/v1")
@click.option("--model", default="reasoning")
@click.option("--panel", default=3, type=int)
@click.option("--max-tokens", default=24000, type=int)
@click.option("--output-dir", default="results/fusion-reasoning")
def main(base_url, model, panel, max_tokens, output_dir):
    client = OpenAI(base_url=base_url, api_key="not-needed")
    tasks = []
    for f in sorted(glob.glob(os.path.join(TASKS_DIR, "*.yaml"))):
        for t in yaml.safe_load(open(f))["tests"]:
            tasks.append(t)
    print(f"fusion-reasoning self-MoA: panel={panel}@T0.7(think) + judge@T0 | {len(tasks)} tasks", flush=True)
    results, tot = [], 0.0
    for i, t in enumerate(tasks, 1):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=panel) as ex:
            cands = list(ex.map(lambda _: gen(client, model, t["prompt"], 0.7, max_tokens), range(panel)))
        cand_block = "\n\n".join(f"## Candidate {j+1}\n{c[:4000]}" for j, c in enumerate(cands))
        agg = gen(client, model, AGG_TMPL.format(n=panel, problem=t["prompt"], cands=cand_block), 0.0, max_tokens)
        sc = grade(t, agg or "")
        tot += sc
        results.append({"task_id": t["id"], "avg_score": sc, "duration_s": round(time.time() - t0, 1)})
        print(f"  [{i}/{len(tasks)}] {t['id']:<24} score={sc:.2f}  {round(time.time()-t0)}s", flush=True)
    n = len(tasks)
    print(f"\n=== FUSION-REASONING: mean {tot/n:.3f} (n={n}) ===", flush=True)
    print(f"=== SOLO ThinkingCap baseline: 0.880 ===", flush=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"model": f"fusion-reasoning(panel={panel})", "tasks": results, "summary": {"mean_score": tot/n, "n": n}},
              open(os.path.join(output_dir, "fusion_reasoning_results.json"), "w"), indent=1)
    print(f"written -> {output_dir}/fusion_reasoning_results.json", flush=True)

if __name__ == "__main__":
    main()
