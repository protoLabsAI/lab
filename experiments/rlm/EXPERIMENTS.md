# RLM iteration log

Running record of "what we changed → what moved." Stable benchmark below; one
row per iteration. Goal: fastest + most accurate RLM using our Qwen stack.

## Locked benchmark — Q3-5

Same 5 LoCoDiff Q3 tasks every time. Q3 was 0/5 in the M0 baseline, so it's
where the scaffold actually breaks — the right place to watch the needle.

```
qdrant_lib_segment_src_index_field_index_numeric_index_mutable_numeric_index.rs (Q2 117KB)
ghostty_src_Command.zig
ghostty_src_renderer_Thread.zig
ghostty_src_inspector_Inspector.zig
ghostty_src_cli_args.zig
aider_aider_sendchat.py
```

(Picked by `--bucket Q3 --n 5` from the canonical `locodiff-250425/prompts/`.)

Reproduce:

```bash
set -a; source ~/.proto/.env; set +a
export GATEWAY_API_KEY="$LITELLM_API_KEY"
uv run python experiments/rlm/eval/run_locodiff.py \
  --n 5 --concurrency 2 --bucket Q3 \
  --max-steps 50 --max-tokens 400000 --max-wall 600
```

Models (constant across iterations):
- Planner: `protolabs/smart` (Qwen3.6-27B-FP8, thinking, 262K)
- Leaf: `protolabs/fast` (Qwen3.6-35B-A3B-FP8 heretic, 131K)
- Both via gateway → Langfuse traces.

## Iterations

### M0 (2026-05-02 17:24) — baseline

**Config:** `max_steps=24, max_tokens=200K, max_wall=600s, planner_max_tokens=8192,
leaf_max_tokens=2048, planner_temp=0.0, leaf_temp=0.1, no per-call timeout`

**Result:** 0/5 Q3.

| Task | Result | Steps | Tokens | Wall | Leaf |
|---|---|---:|---:|---:|---:|
| sendchat.py        | FAIL[max_steps]  | 14 | 142,003 | 622s | 0 |
| Command.zig        | FAIL[max_steps]  | 22 | 200,116 | 629s | 0 |
| Thread.zig         | FAIL[max_steps]  | 22 | 209,129 | 530s | 0 |
| args.zig           | FAIL[error]      |  8 |  40,220 | 414s | 0 |
| Inspector.zig      | FAIL[max_steps]  |  9 | 115,291 | 782s | 0 |

**Findings:**
- Planner never used `RLM_MAP` even once (`leaf_calls=0` everywhere).
- All `error` cases were "neither code nor FINAL" — planner gave up mid-thought.
- Most max_steps cases were the planner iterating on its own diff parser.

### M0+1 (2026-05-02 18:55) — RLM_MAP-teaching prompt + 32K planner tokens

**Config:** prompt revised with worked `RLM_MAP` example for diff-applier
pattern + "if Python fails twice, switch" rule. `planner_max_tokens=32K`,
`leaf_max_tokens=32K`.

**Result:** 1/5 Q3 — needle moved by 1.

| Task | Result | Steps | Tokens | Wall | Leaf |
|---|---|---:|---:|---:|---:|
| sendchat.py        | FAIL[exception]  |  0 |       0 | 2247s | 0 |
| Command.zig        | **PASS**         | 10 |  56,578 |  191s | 0 |
| Thread.zig         | FAIL[max_steps]  | 11 |  88,109 |  878s | 0 |
| args.zig           | FAIL[max_steps]  | 17 | 126,017 |  835s | 0 |
| Inspector.zig      | FAIL[max_steps]  | 24 | 251,907 | 1716s | 0 |

**Findings:**
- **Still zero leaf calls.** Prompt nudging alone didn't flip behavior — planner
  prefers Python every time, even with worked example.
- **32K max_tokens makes single turns explode**: Inspector.zig had ONE planner
  turn that took 1227s (20 min) of pure thinking. With 24 of those, total wall
  hit 1716s — well over the 600s wall budget, but the budget was never enforced
  because (a) it's only checked between super-steps, not during a call, and
  (b) the trajectory mislabeled it as `max_steps`.
- **One full-session hang**: sendchat.py — first planner call hung 2247s with
  zero tokens emitted. Gateway/vLLM didn't return at all. No per-call timeout
  was set, so client just waited.
- **vLLM doesn't promptly abort on disconnect**: when we killed the python
  client, vLLM kept generating for ~10 min.

### Iteration 2 (2026-05-02 19:46) — orchestrator gaps fixed

**Changes from M0+1:**
1. **Per-call timeouts**: planner=180s, leaf=60s. OpenAI SDK raises
   `APITimeoutError` on overrun; httpx closes the connection so vLLM gets
   the disconnect signal.
2. **Truthful terminate-reason labels**: `max_steps`, `max_tokens`, `max_wall`,
   `planner_timeout`, `planner_error`, `final`, `error`. No more silent
   "max_steps" fallback. Required reading the trajectory honestly.
3. **Planner max_tokens 32K → 16K**: middle ground. 8K was too low (truncated
   mid-fence in M0). 32K let single turns dominate wall. 16K bounds turn cost
   to ~5-7 min worst case.
4. **`max_tokens` budget bumped to 400K** (default + CLI). 200K was binding in
   M0+1 on Inspector.zig.

**Result:** 1/5 Q3 — same number as M0+1, but the bug fixes did real work.

| Task | Result | Steps | Tokens | Wall | Leaf |
|---|---|---:|---:|---:|---:|
| sendchat.py        | FAIL[max_wall]   | 11 |  75,543 |  844s | 0 |
| Command.zig        | FAIL[max_wall]   | 12 |  71,955 |  834s | 0 |
| Thread.zig         | **PASS**         |  8 |  39,701 |   86s | 0 |
| args.zig           | FAIL[max_wall]   | 12 |  92,109 |  819s | 0 |
| Inspector.zig      | FAIL[max_wall]   | 19 | 210,214 | 1003s | 0 |

**Findings:**
- ✅ **Per-call timeout works**: sendchat went from "0 steps, 2247s exception"
  → "11 steps, 844s max_wall". Silent infinite hang is dead.
- ✅ **Truthful labels work**: every failure now correctly labeled `max_wall`.
  Wall is the real constraint, not step count. Earlier diagnoses were wrong.
- ❌ **Still 0 leaf calls.** Prompt-only nudging won't flip the planner —
  even with the worked example, it defaults to "I can solve in pure Python"
  every time. Consistent with the paper's argument that you need to *train*
  a model to do RLM well.
- ⚠️ Different task passed than M0+1 (Thread.zig instead of Command.zig);
  both at temp=0 so it's not RNG — different planner strategies emerged.
- Inspector.zig and sendchat.py were *making progress* when wall fired
  (210K and 75K leaf tokens respectively, mid-application of diffs).

### Heretic config fix (2026-05-02 21:24) — root cause of the leaf `<think>` leak

**The bug**: vllm-fast.service was missing `--reasoning-parser qwen3`.
- Heretic (and the official 35B-A3B-FP8) has trained-in always-thinks behavior.
- The nothink chat template's "Fix 19" only suppresses thinking on tool-call
  turns. RLM leaf calls have no tools → thinking fires every time.
- Without `--reasoning-parser qwen3`, vLLM has no logic to route `<think>`
  blocks into `reasoning_content`. They land in `content` verbatim.
- Our `LeafClient._strip_thinking` cleaned them up before the planner saw
  them, but downstream consumers (including the protoCLI release-notes
  generator) saw the leak in production.

**The fix**: added `--reasoning-parser qwen3` to `vllm-fast.service` (matching
the existing `vllm.service` for smart). Restarted, ~44s warm-up.

**Validation** (post-restart probes):
- Direct :8002, simple prompt → `content: 'Hi, how are you doing today?'` ✓
- Direct :8002, thinking-bait + 600 tok cap → `content: None` (vLLM #40528,
  unfixed: when model truncates mid-`<think>` the parser drops everything
  to `reasoning_content`, content becomes None). Acceptable: gateway-side
  `thinking_normalizer.py` salvages this for non-streaming, and our
  LeafClient already has the fallback chain `content or reasoning or
  reasoning_content`.
- **Gateway, thinking-bait → `content: 'To calculate 17×23 ...'` ✓ clean,
  `reasoning_content: '\\n\\n'`.** Only path that matters for RLM.

CLAUDE.md updated. Backup at `/etc/systemd/system/vllm-fast.service.bak.<epoch>`.

### Iteration 3 (2026-05-02 21:48) — clean heretic leaf

**Result:** 1/5 Q3 — same as iter-2 BUT loop is materially tighter.

| Task | Result | Steps | Tokens | Wall | Leaf |
|---|---|---:|---:|---:|---:|
| sendchat.py        | FAIL[max_wall]         | 14 |  97,186 |  759s | 0 |
| Command.zig        | **PASS**               |  5 |  16,844 |  111s | 0 |
| Thread.zig         | FAIL[final]            |  9 |  48,330 |  411s | 0 |
| args.zig           | FAIL[max_wall]         | 12 |  89,661 |  604s | 0 |
| Inspector.zig      | FAIL[planner_timeout]  |  7 |  27,709 |  754s | 0 |

**Findings:**
- ✅ Total wall **1363s vs iter-2's 1848s = 26% faster.** Same accuracy
  but the loop is tighter — plausibly cleaner leaf paths through gateway,
  though leaf_calls=0 everywhere so the win is non-deterministic noise on
  planner side.
- ✅ **Planner_timeout label fired correctly** on Inspector.zig — single
  planner turn exceeded 180s and we cut it off. New failure mode now
  observable, no more silent hangs.
- ✅ Command.zig is back to PASS but in 5 steps / 111s (was 12 steps /
  834s in iter-2). Same task, same temp=0, dramatically different planner
  trajectory. Confirms planner non-determinism we hadn't budgeted for.
- ⚠️ Thread.zig: `terminated_reason=final` but FAIL — planner emitted FINAL
  with the WRONG answer. Quality failure, not budget. Worth a per-trajectory
  look later.
- ❌ Still 0 leaf calls. Not surprising given iter-3 was a regression test
  for the heretic config, not a behavioral change.

**M0 final baseline locked:** **1/5 Q3 (20%) on the M0+1 prompt with
gateway routing, clean leaf, truthful labels, per-call timeouts.** This is
the number Phase 1 of [PROPOSAL.md](PROPOSAL.md) compares against.

**Single changed variable from iter-2:** leaf model now has
`--reasoning-parser qwen3` server-side, so leaf responses come back with
clean `content` instead of `<think>`-laden text.

All other knobs identical:
- planner=protolabs/smart (unchanged)
- leaf=protolabs/fast (now reasoning-parser configured)
- max_steps=50, max_tokens=400K, max_wall=600s
- planner_timeout=180s, leaf_timeout=60s
- planner_max_tokens=16K, leaf_max_tokens=32K
- RLM_MAP-teaching prompt, planner temp=0

**Hypothesis:** if the planner *did* call leaf in iter-2 (it didn't), the
leaf would have wasted think tokens that our client stripped. Now the
leaf is honest about cost. **But since iter-2 had 0 leaf calls, this
iteration is mostly a regression test** — confirming we didn't break
anything by changing the unit. If the planner still ignores `RLM_MAP`,
the result should mirror iter-2 (1/5 with Thread.zig passing).

If iter-3 differs materially from iter-2 with no other changes, that's
informative on its own (likely planner non-determinism we hadn't budgeted
for).

## Open questions

- Will any prompt-only change get the planner to use `RLM_MAP`, or do we need
  to post-train on RLM trajectories (M4 in PLAN.md)?
- Should we make the leaf the planner instead? Heretic uses the
  `qwen3.5-tool-calling-nothink.jinja` template — much shorter `<think>` blocks.
  Trade quality of plan for wall speed.
- Sandbox primitives: should we just preinstall `unidiff` so the planner doesn't
  have to write its own diff parser? Compromises the paper's pure-orchestration
  thesis but might be the right call for *this* benchmark.
