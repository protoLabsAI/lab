"""Self-MoA (fusion) over the coder lane on LiveCodeBench — bench vs solo.

Replicates the gateway's protolabs/fusion-coder recipe locally against the live
coder lane: N candidates @ T=0.7, then a T=0 judge synthesizes the final solution.
Same problem set + grader as run_livecodebench, so results compare directly to the
solo coder run (30% solve / 0.537 mean).

Usage:
  python -m runners.fusion_lcb --base-url http://127.0.0.1:8032/v1 --model coder \
      --panel 3 --limit 30 --output-dir results/fusion-lcb
"""
from __future__ import annotations
import json, os, time, click
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
from runners.run_livecodebench import load_problems, build_prompt, grade_problem
from graders.code_exec import extract_code

AGG_TMPL = """You are given a competitive-programming problem and {n} candidate solutions from independent attempts. Some may be buggy or incomplete. Study them, reason about correctness and edge cases, then produce ONE final, correct, complete solution.

# Problem
{problem}

# Candidate solutions
{cands}

Respond with ONLY the final solution as a single ```python code block — no commentary."""

def gen(client, model, prompt, temp, max_tokens):
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=temp, top_p=0.95, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20, "min_p": 0.0},
        )
        m = r.choices[0].message
        return (m.content or "") or (getattr(m, "reasoning_content", None) or "")
    except Exception as e:
        return f"__ERR__ {e}"

@click.command()
@click.option("--base-url", default="http://127.0.0.1:8032/v1")
@click.option("--model", default="coder")
@click.option("--panel", default=3, type=int, help="N candidates @ T=0.7")
@click.option("--limit", default=30, type=int)
@click.option("--version-tag", default="release_v6")
@click.option("--min-date", default="2025-01-01")
@click.option("--max-tokens", default=16384, type=int)
@click.option("--output-dir", default="results/fusion-lcb")
def main(base_url, model, panel, limit, version_tag, min_date, max_tokens, output_dir):
    client = OpenAI(base_url=base_url, api_key="not-needed")
    problems = load_problems(version_tag, min_date, ["hard"], limit)
    print(f"fusion-coder self-MoA: panel={panel}@T0.7 + judge@T0 | {len(problems)} problems")
    results, solved, tot = [], 0, 0.0
    for i, p in enumerate(problems, 1):
        t0 = time.time()
        prompt = build_prompt(p)
        with ThreadPoolExecutor(max_workers=panel) as ex:
            cands = list(ex.map(lambda _: gen(client, model, prompt, 0.7, max_tokens), range(panel)))
        cand_block = "\n\n".join(f"## Candidate {j+1}\n{c[:6000]}" for j, c in enumerate(cands))
        agg = gen(client, model, AGG_TMPL.format(n=panel, problem=prompt, cands=cand_block), 0.0, max_tokens)
        code = extract_code(agg or "")
        g = grade_problem(p, agg or "", 20, 6, 90.0)
        sc = g.get("score", 0.0); ok = g.get("passed", 0) == g.get("total", 1) and g.get("total", 0) > 0
        solved += 1 if ok else 0; tot += sc
        results.append({"task_id": p.get("question_id"), "title": p.get("question_title"),
                        "avg_score": sc, "passed": g.get("passed"), "total": g.get("total"),
                        "error": g.get("error"), "duration_s": round(time.time() - t0, 1)})
        print(f"  [{i}/{len(problems)}] {str(p.get('question_title'))[:40]:<40} {g.get('passed')}/{g.get('total')} score={sc:.2f} {'SOLVED' if ok else ''} {round(time.time()-t0)}s")
    n = len(problems)
    summary = {"mean_score": tot / n, "solve_rate": solved / n, "n": n, "panel": panel}
    print(f"\n=== FUSION-CODER: mean {summary['mean_score']:.3f} | solve {summary['solve_rate']:.0%} ({solved}/{n}) ===")
    print(f"=== SOLO coder baseline: mean 0.537 | solve 30% (9/30) ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"model": f"fusion-coder(panel={panel})", "problems": results, "summary": summary},
              open(os.path.join(output_dir, "fusion_lcb_results.json"), "w"), indent=1)
    print(f"written -> {output_dir}/fusion_lcb_results.json")

if __name__ == "__main__":
    main()
