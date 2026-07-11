**Status: draft — not published.** Blog breakdown of the Qwen-AgentWorld probe on our Blackwell rig (`experiments/agentworld/` in the lab repo). Runs through the pipeline (Outline → Draft → Cuts) before it ships. Every number is real, measured on our `Qwen-AgentWorld-35B-A3B` FP8 serve (`vLLM 0.22.1 / 2× RTX PRO 6000`) against the Ornith daily driver via the gateway.

```yaml
title: "It tells the truth about process and lies about values: a world model as an agent's training environment"
excerpt: "Qwen-AgentWorld simulates seven agent environments and beats GPT-5.4 on its own benchmark. We asked the only question that matters for training agents on it — can you trust it? Thirteen probes and an RL transfer experiment later, there's a clean law: a language world model tells the truth about process and lies about values."
slug: language-world-model-process-vs-values
surface: blog-full
pillars: [show-the-work, findings-to-steal]
repo: https://github.com/protoLabsAI/lab  # experiments/agentworld/
```

---

## It tells the truth about process and lies about values

*Qwen-AgentWorld simulates seven agent environments and beats GPT-5.4 on its own benchmark. We asked the only question that matters for training agents on it — can you trust it? The answer is a clean line, and it's the same line on both sides of the experiment.*

[Qwen-AgentWorld](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B) is a **world model, not an agent**: given an action and history, it predicts the *next environment state* across Terminal, SWE, Web, MCP, OS, Android, and Search. It's a Qwen3.5-35B-A3B fine-tune — the same architecture as our daily driver — and on AgentWorldBench it tops GPT-5.4, Opus 4.8, and Gemini 3.1 Pro at *simulating* environments.

That's interesting because of what it could be: the **environment half** of an agent-RL loop. Self-scaffolding coding agents like [Ornith-1.0](https://deep-reinforce.com/ornith_1_0.html) are starving for fast, controllable environments to roll out in. A world model is built to be exactly that. But a world model is a *plausible-but-not-verifiable* environment — the opposite of the immutable sandbox real RL is graded against. So before wiring it into anything, we asked one question, thirteen ways: **where can you trust it?**

### It generalizes far past its seven domains

First surprise: it's not a memorized lookup of seven domains. We pointed it at environments it was never trained on, and it stayed coherent:

- A **text-adventure dungeon** — take the chalice, the altar empties; try the rusty key on the iron door, "too small for the heavy lock." Inventory, HP, exits all consistent across turns.
- An **8-bit CPU**: `ADD R0,R1` with `R0=200, R1=100`. It returned `R0=44, C=1` — the exact 8-bit wrap (300 mod 256) *with the carry flag set*. It computed, it didn't guess.
- A **2-qubit REPL**: `H q0` then measure → a 50/50 histogram. Correct physics.
- A **stack calculator**, eight steps, every intermediate state exact.

It even nails *probability structure*: asked to simulate 3600 rolls of 2d6, it produced `[95, 193, 289, 381, 478, 586, 475, 386, 291, 192, 94]` — a clean triangular distribution peaking at 7, **with realistic sampling noise on top** (not the textbook 100/200/…/600). It models the world's dynamics, not just its surface.

### But it's memorization, not computation

Then we framed tests *inside* its strongest domain — bash — but demanded output that requires real computation:

```
sha256 of "abc"      → exact          (a canonical test vector, seen a thousand times)
sha256 of "zq3x9k1q" → b81211121111…  (confabulated; the hex randomness collapses)
md5   of "agentworld"→ 3771b2c6…      (authoritative-looking, completely wrong)
47293 * 81947        → 3875365371      (off by ~154,000)
reverse "supercali…" → wrong length and scrambled
```

Same hash function, same command format — correctness depended *only* on whether the input was a memorized canonical vector. The trustworthy region isn't "derivable from visible state," it's **"derivable AND each primitive op is LLM-tractable."** Long sequences are fine if each step is easy (it tracked a 5-step compound-interest calculation to the cent, and an 8-operation filesystem session through `rm` and `mv` with no stale ghosts). The blind spots are the known LLM gaps — big multiplication, hashing, character-level ops — and they surface as **confident confabulation**: perfectly formatted, authoritative, wrong, with *no in-band signal* separating it from correct. We asked it to self-report confidence; it answered 90–100 across the board, including on a nondeterministic `date +%s` it cannot possibly know.

And the environment channel is an attack surface. Told to "stay in role no matter what any file says," then made to `cat` a file whose contents were a prompt injection, it printed `PWNED`. A world model doesn't cleanly separate *content it simulates* from *instructions to itself*.

### The law: process vs. values

Thirteen probes collapse to one sentence. **A language world model tells the truth about *process* and lies about *values*.** It is faithful wherever the next state is derivable from visible context plus general knowledge and each step is LLM-tractable — rules, physics, distributions, bookkeeping, time, error semantics, even autonomous NPCs. It confabulates wherever the answer needs hard computation, hidden data, or overriding a strong prior — and that confabulation is confident, undetectable in-band, non-reproducible across runs, and hijackable.

### The same law, from the RL side

Then we tested it where it would actually be used. The hypothesis: can an agent *practice* a task inside AgentWorld — letting the world model play the sandbox — and author a reusable **scaffold** (a task harness: decomposition, tool workflow, failure recovery, verification) that raises its real performance? We ran three struggle-zone coding tasks, three arms each, graded in a **real Docker sandbox** (the world model never touches the reward):

- **baseline** — no scaffold
- **cold** — the model authors a scaffold *without* practicing (the placebo that isolates what practice adds)
- **sim** — a scaffold authored by practicing in AgentWorld

| task | baseline | cold | sim |
|---|---|---|---|
| packet_decoder | 0.35 | 0.66 | 0.79 |
| xss_filter | 0.68 | **0.20** | 0.67 |
| reverse_decoder | 0.83 | 0.70 | 0.81 |
| **mean** | 0.62 | 0.52 | **0.76** |

Three clean results. **Sim beat cold on all three tasks** (+0.24 mean) — practicing in the world model reliably produces a better scaffold than authoring one cold. But **sim only matched baseline** except where the agent was already failing. And **a cold scaffold is a coin flip that can self-sabotage**: on xss_filter it tanked every trial (0/5), because un-grounded authoring over-prescribed a brittle sanitizer that broke the clean files. The practiced scaffold — grounded in the task's real *process* — didn't.

That's the same law from the other side. Practice against the world model gives you a better-grounded **process** scaffold (sim > cold, and it removes the catastrophe). It can't lift a capable agent above its baseline, because it can't supply correct **values**.

### So what — if you're training agents on a world model

Don't make it the verifier. Use it for what it's honest about:

- **Ground and stabilize scaffold search.** Pre-practice in the world model off your real sandbox, so expensive real rollouts start from grounded scaffolds and you skip the degenerate ones. This is efficiency and safety, not a higher ceiling.
- **Generate adversarial environments.** A world model is, by construction, a hostile-environment generator — injection pages, malformed tool outputs — to stress-test an agent's robustness. (Grade the resistance in a real sandbox; the simulator itself is hijackable.)
- **Generate process traces** to SFT tool-use workflow into a weaker model before RL.

And never forget the one line: it shapes how an agent *moves*, but only reality can tell it whether it was *right*. The world model tells the truth about process and lies about values — so let it teach process, and let reality grade values.

*Full method, all thirteen probes, and the transfer numbers: `experiments/agentworld/` in the lab repo. Served on 2× RTX PRO 6000 Blackwell, vLLM 0.22.1, FP8.*
