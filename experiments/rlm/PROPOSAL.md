# PROPOSAL — Compounding Recursive Language Models

> A library-learning extension to RLM that grows a typed, gateway-traced
> catalog of LangGraph subgraphs as it solves problems. Built on DSPy,
> seeded from MIT's RLM trajectories, published as portable HuggingFace
> datasets per domain.

**Status:** draft, supersedes [PLAN.md](PLAN.md)
**Author:** protoLabs lab, 2026-05-02
**Targets:** workshop paper Q3 2026; first published library `protoLabsAI/rlm-library-coding` Q4 2026

---

## 0. TL;DR

We're not building "RLM but ours." We're synthesizing six existing ideas
into one productionized stack and shipping the genuinely missing piece
— a **typed, type-checked, gateway-observable subgraph library that
compounds across queries**.

| Ingredient | Source | What we use |
|---|---|---|
| Recursion over decomposed context | [Zhang/Kraska/Khattab — RLM](https://arxiv.org/abs/2512.24601) | The core inference paradigm + reference impl |
| Skill-library lifelong learning | [Wang et al. — Voyager](https://arxiv.org/abs/2305.16291) | Cross-query compounding + auto-curriculum from real usage |
| Library compression / refactoring | [Bowers — Stitch (POPL '23)](https://dl.acm.org/doi/10.1145/3571234) | Periodic compaction so the catalog doesn't bloat |
| LM-guided library learning | [Grand et al. — LILO (ICLR '24)](https://arxiv.org/abs/2310.19791) | Synthesis loop + interpretable abstractions |
| Programming, not prompting | [DSPy](https://dspy.ai/) (Stanford) | Signatures, modules, GEPA prompt optimizer |
| Multi-role decomposition | [ROMA](https://github.com/sentient-agi/ROMA) | Atomizer / Planner / Executor / Aggregator inspiration |
| Memory typing | [Letta (MemGPT)](https://github.com/letta-ai/letta) | Episodic / semantic / procedural memory taxonomy |

The novelty we ship: **type-checked LangGraph subgraphs as the unit of
compounding skill, gateway-traced for production observability, published
as per-domain HF datasets, with a cold-start of hand-curated nodes.**

That's the whole sentence. The rest of this document is the strategy and
execution plan to deliver it.

---

## 1. Antagonistic review of where we were heading

Before committing, I ran an adversarial pass on yesterday's "library-learning
RLM" sketch. Of twelve criticisms, **seven hit hard enough to change the plan**:

| # | Criticism | Verdict | Adjustment |
|---|---|---|---|
| 1 | Library learning isn't new — Voyager / DreamCoder / LILO already exist | **Valid** | Stop pretending novelty; cite explicitly; differentiate on production / publishing axis |
| 2 | Type signatures over arbitrary Python objects don't work in practice | **Partial** | Constrain library node I/O to common Pydantic-friendly types; reject Any |
| 3 | Synthesis from primitive vocabulary won't beat direct code-gen | **Defensible** | Keep primitives small; let LM compose; A/B vs free-form codegen in Phase 2 |
| 4 | Our base 27B might be too weak for any of this to work | **Valid** | Phase 1 must include: download `mit-oasys/rlm-qwen3-8b-v0.1`, A/B as planner |
| 5 | LoCoDiff is the wrong benchmark for library transfer | **Valid** | Add 2 benchmarks where same-class problems repeat: lab-repo audit + LongBench v2 |
| 6 | 5 phases is too slow; momentum dies | **Valid** | Compressed to 2 phases + ongoing curation |
| 7 | ROMA already exists with DSPy + GEPA + SOTA; why build? | **Partial** | Don't replace ROMA — *add* the library layer; integrate-or-replace decision in Phase 1 |
| 8 | Library + weak planner = 0 + 0 = 0 | **Valid** | Phase 1 includes both axes: library + post-trained planner. Either alone is the null hypothesis |
| 9 | "Publishable artifact" is brand-thinking polluting engineering | **Valid** | Brand fit is a real constraint, not a goal. Acknowledge openly. |
| 10 | Subgraphs vs sub-agents is an arbitrary taste call | **Defensible** | Subgraphs win on auditability + Langfuse traceability. Stand by it. |
| 11 | Why not just call Sonnet 4.5? | **Valid** | Honest answer: we serve customers who can't / won't send data outside their network. Document this as our positioning. |
| 12 | Collecting 0-leaf-call trajectories teaches "don't recurse" | **Valid** | Don't SFT on our own bad trajectories. Use MIT's. Phase 2 trajectories are filtered by leaf_calls > 0 + correct. |

Net effect: the plan got tighter, more honest, less ambitious about
inventing-from-scratch, more focused on the productionization gap that's
genuinely missing in the research stack.

---

## 2. Prior art (the honest accounting)

### 2.1 The closest existing system — LILO (ICLR 2024)

[LILO: Learning Interpretable Libraries by Compressing and Documenting Code](https://arxiv.org/abs/2310.19791)
combines LLM-guided program synthesis with Stitch's symbolic refactoring to
build an interpretable, growing library of code abstractions. It outperforms
DreamCoder on synthesis, runs Stitch in seconds (not days), and produces
documented library entries.

**What LILO has that we'd be re-inventing:** the LLM-synthesis + symbolic-refactor loop.

**What LILO doesn't have that we'd add:**
- Production observability (Langfuse trace per library invocation, success metadata over time)
- Heterogeneous compute backends (small leaf model + large planner via gateway)
- Multi-domain library publishing as artifact
- Recursive sub-call as a primitive (LILO is single-LM)
- Cold-start curation (LILO is bootstrap-from-scratch)

**Verdict:** we should fork or adapt LILO's loop, not re-invent it. If the
authors' code is usable, we cite, fork, and extend. If not, we re-implement
the loop in DSPy and cite.

### 2.2 Voyager — what worked, what was a hidden gift

[Voyager (Wang 2023)](https://arxiv.org/abs/2305.16291) showed:
- Skill libraries DO compound: 3.3× more items, 15.3× faster milestones, transfer to new worlds
- Auto-curriculum + iterative self-verification + skill library is the right loop

**The hidden gift:** Minecraft is a perfect simulator — instant errors,
clean reset, no side effects, binary success. Our REPL has none of these.
**The verifier is the engineering hardest part of our system**, not the
catalog. Plan accordingly.

### 2.3 DreamCoder & Stitch — library compression matters

DreamCoder showed library learning works. Stitch made it 1,000-10,000× faster.
We should use Stitch (or its principles) for periodic compaction. Without
compression, the library will balloon into a useless graveyard within months.
[`mlb2251/stitch`](https://github.com/mlb2251/stitch) is the reference impl.

### 2.4 DSPy — we use it, we don't compete with it

[DSPy](https://dspy.ai/) is the framework for "programming, not prompting."
ROMA already integrates it. We should:
- Wrap planner / synthesizer / verifier as DSPy modules with signatures
- Use **GEPA** to optimize their prompts on real trajectories
- Use `BootstrapFinetune` if we ever do model-side training

This eliminates a huge amount of prompt-engineering work and gives us
an upgrade path that doesn't require manual tuning.

### 2.5 ROMA — the role separation we partially borrow

[ROMA](https://github.com/sentient-agi/ROMA) hits SEAL-0 45.6%, FRAMES 81.7%,
SimpleQA 93.9% with **Atomizer / Planner / Executor / Aggregator** roles.
We adopt the role taxonomy as inspiration but don't adopt the rigidity.
Our planner is LM-decided (RLM-style); the roles are *hints* in the prompt
and *organization* in the trajectory schema, not hard-coded gates.

**Open decision (Phase 1):** fork ROMA + add library layer, OR build on
top of MIT's `rlm` reference impl + add ROMA-style role hints. Decided after
benchmarking both as planners on Q3-5.

### 2.6 Letta / MemGPT — memory taxonomy we steal

Letta's three memory types map cleanly:
- **Episodic** = our trajectories (stored, queryable by intent)
- **Semantic** = derived facts about library nodes (success rate, usage count, tags)
- **Procedural** = the library nodes themselves (executable code)

We don't need Letta the runtime. We need its taxonomy as an organizing
principle for what we persist.

### 2.7 The MIT post-train as a starting point

[`mit-oasys/rlm-qwen3-8b-v0.1`](https://huggingface.co/mit-oasys/rlm-qwen3-8b-v0.1) is
fine-tuned on 1,000 filtered trajectories of `Qwen3-Coder-480B-A35B`-as-RLM
on LongBenchPro. **+28.3% over base 8B.** This model exists *today*. We
should use it as our planner baseline before doing any post-training of our own.

---

## 3. What's actually novel

After honest crediting of prior art, three things remain that nobody has shipped:

### 3.1 Type-checked, gateway-traced library nodes

Every library entry is a LangGraph subgraph with **Pydantic input/output
schemas**. Retrieval combines intent-embedding similarity *and* type
compatibility — you can reject a candidate as a hard miss if its types
don't fit the task. This is materially stronger than RAG-style "find similar
text" used by every memory-of-LM-skills system to date.

Every invocation routes through our LiteLLM gateway → Langfuse captures
the trace → the catalog gets updated success metadata atomically. This is
the productionization that academic library-learning systems lack.

### 3.2 Multi-domain libraries as portable artifacts

We publish per-domain libraries to HuggingFace as datasets:
- `protoLabsAI/rlm-library-coding` (LoCoDiff, codebase audit)
- `protoLabsAI/rlm-library-research` (LongBench v2, multi-doc QA)
- `protoLabsAI/rlm-library-finance` (TBD)

Each is a versioned, diffable, tested catalog. Organizations can fork,
extend, and publish their own. Companies generally **can't** share fine-tuned
models for IP reasons; they **can** share libraries that are made of code +
metadata + scoring evidence.

This is the protoLabs.studio brand fit, but it's also a real engineering
artifact: institutional knowledge made auditable.

### 3.3 Cold-start with hand-curated nodes + LM-grown extensions

Voyager bootstrapped from zero. LILO bootstrapped from a curriculum.
We bootstrap from **5-10 hand-built nodes** designed for our domain
(coding/research/finance). This sidesteps the "library is empty for the
first 100 queries" problem that kills demos.

LM synthesis fires only when retrieval misses. Hand-built nodes are
versioned, tested, documented. LM-built nodes are gated by a verifier
before commit. Both end up in the same catalog with the same schema.

---

## 4. System architecture

```
                        ┌──────────────────────────────────┐
   user query  ─────►   │            ROOT PLANNER (DSPy)    │
   + context           │   inputs: query, ctx_meta, top-K  │
                        │   library candidates              │
                        │   output: action ∈ {INVOKE,       │
                        │     COMBINE, SYNTHESIZE,          │
                        │     DIRECT_SOLVE}                 │
                        └────────────┬─────────────────────┘
                                     │
              ┌──────────────────────┼─────────────────────────┐
              │                      │                         │
              ▼                      ▼                         ▼
     ┌─────────────┐       ┌──────────────┐         ┌─────────────────┐
     │  LIBRARY    │       │  SYNTHESIZER │         │  DIRECT_SOLVE   │
     │  RETRIEVE   │       │  (DSPy mod)  │         │  (current RLM   │
     │             │       │              │         │   in-context)   │
     │  FAISS over │       │  composes    │         └─────────────────┘
     │  intents +  │       │  primitives: │
     │  type-match │       │  exec, leaf, │
     │  filter     │       │  partition,  │
     │             │       │  agg, regex, │
     │             │       │  summarize   │
     └──────┬──────┘       └───────┬──────┘
            │                      │
            ▼                      ▼
     ┌─────────────┐       ┌──────────────┐
     │  EXECUTE    │       │  VERIFY      │
     │  (chosen    │       │  (Docker     │
     │   subgraph  │       │   sandbox,   │
     │   in        │       │   score,     │
     │   sandbox)  │       │   gate)      │
     └──────┬──────┘       └───────┬──────┘
            │                      │
            ▼                      ▼ (if score ≥ threshold,
     ┌──────────────────────────┐    no near-dup, lint pass)
     │  TRAJECTORY              │  ─►  COMMIT to library
     │  + Langfuse trace        │
     │  + library success       │
     │    metadata update       │
     └──────────────────────────┘
                                       offline:
                                       ┌─────────────────────┐
                                       │  CURATOR            │
                                       │  - Stitch refactor  │
                                       │  - LRU prune        │
                                       │  - cosine dedupe    │
                                       │  - HF publish       │
                                       └─────────────────────┘
```

### Key decisions baked into the architecture

| Decision | Rationale |
|---|---|
| Planner emits one of 4 actions | Smaller, more answerable choice space than "decompose freely" |
| DSPy modules everywhere | Free GEPA optimization; portable signatures; testability |
| Library retrieval BEFORE planner sees task | Planner choice is informed by what's available |
| Synthesizer composes primitives, not free-form Python | Bounded search space; primitives are auditable |
| Verifier in Docker sandbox | Safety-critical for LM-generated code commit |
| Curator runs offline | Compaction shouldn't block the hot path |
| Same schema for hand + LM nodes | One catalog, one retrieval path, one quality bar |

---

## 5. Plan (compressed to 2 phases + ongoing)

### Phase 0 — In flight, finishing now

Iteration 3 against Q3-5 baseline (clean heretic leaf). Logs the final
"vanilla RLM scaffold" number we benchmark Phase 1 against. Already running.

**Exit:** numbers in `EXPERIMENTS.md`.

### Phase 1 — Retrieval-only library + planner A/B (weeks 1-2)

**Build:**
- Annotate `Trajectory` with per-turn `intent` + Pydantic input/output signature
- Wrap planner / leaf / sandbox primitives as DSPy modules with signatures
- Hand-build 5 library nodes targeting our 3 benchmarks:
  - `git_diff_applier` (LoCoDiff)
  - `multi_doc_aggregator` (LongBench v2 multi-doc QA)
  - `codebase_grep` (lab-repo audit)
  - `regex_extract` (general utility)
  - `chunk_summarize` (general utility)
- FAISS index over node intent embeddings; retrieval also filters by type sig
- Modified planner prompt: includes top-K candidates with intent + sig + success rate
- Planner emits `INVOKE(node_id, args)` or `DIRECT_SOLVE`

**A/B planners on the SAME 3 benchmarks:**
- `protolabs/smart` (current)
- `mit-oasys/rlm-qwen3-8b-v0.1` (downloaded; served via vLLM)
- `protolabs/smart` + GEPA-optimized prompts

**Kill criteria** (any of):
- Library retrieval doesn't help on lab-repo audit (the benchmark designed FOR retrieval)
- MIT planner + library doesn't beat MIT planner alone
- DSPy + GEPA gives no measurable gain on planner
- Combined: **+0pp on lab-repo audit at Phase 1 end** = library hypothesis is dead, pivot to direct MIT planner adoption only

**Exit:** RESULTS.md with three planner × three benchmark numbers; clear go/no-go on Phase 2.

### Phase 2 — Synthesis + verification + library growth (weeks 3-5)

**Build (only if Phase 1 passes go/no-go):**
- LM synthesizer composes new nodes from a fixed primitive vocabulary
- Per-node Docker sandbox for verification (safety-critical; non-negotiable)
- Verification scoring: solves task? generalizes to ≥1 held-out variant?
- Commit gate: score ≥ 0.7, no cosine-near duplicate, passes safety lint
- Stitch-style periodic compaction (weekly cron)
- DSPy GEPA optimizes planner's INVOKE / SYNTHESIZE / COMBINE / DIRECT_SOLVE selection on collected trajectories

**Kill criteria:**
- Synthesis success rate < 30% (synthesizer can't reliably build working nodes)
- Library doesn't compound (no measurable improvement after 100 queries)
- Verifier false-positive rate > 10% (commits broken nodes)

**Exit:** library has grown by ≥5 LM-generated nodes that survive 50+ queries; benchmarks improve over Phase 1 hand-curated baseline.

### Ongoing — Curation + publication (continuous from Phase 1)

- Weekly: Stitch refactor + cosine dedupe; deprecate low-success nodes
- Monthly: snapshot library to `protoLabsAI/rlm-library-{coding,research}` HF datasets with version tag
- Per-PR: BENCHMARKS.md auto-update with current numbers
- Quarterly: blog draft for protolabs.studio

---

## 6. Benchmarks (locked, three of them)

We pick three so the hypothesis gets stressed from different angles. Same
benchmarks every iteration; **never change the test, only the system.**

| Benchmark | Source | Tests | Why |
|---|---|---|---|
| **LoCoDiff Q3-5** | [AbanteAI/LoCoDiff-bench](https://github.com/AbanteAI/LoCoDiff-bench) | One-shot exact reconstruction; weak transfer expected | Diagnostic: does the library *hurt*? Worst-case test |
| **lab-repo audit** | Custom, 50 questions over `~/dev/lab/` | Same-class queries over same corpus; **strong transfer expected** | Designed FOR retrieval-helps; if this doesn't move, hypothesis is dead |
| **LongBench v2 — multi-doc QA + code-repo subset** | [THUDM/LongBench](https://github.com/THUDM/LongBench) | Multi-doc QA, repo understanding, dialogue history; 503 hard MCQs | Production-grade RLM benchmark; comparable to MIT's claims |

Each iteration runs all three. Results table tracks every config × benchmark
× planner combination. **Lab-repo audit is the make-or-break for the library
hypothesis** — it's a curated set of questions designed so library transfer
should help if it helps anywhere.

### Lab-repo audit construction

50 questions hand-written over `~/dev/lab/`, structured by question type:
- **Locate** (10): "where is X defined?" — should retrieve `codebase_grep` node every time
- **Summarize** (10): "what does directory X do?" — should retrieve `chunk_summarize` + `multi_doc_aggregator`
- **Cross-reference** (10): "what are all places that call function X?" — composes grep + multi-doc-agg
- **Diff-style** (10): "what changed in module X between commit A and B?" — should retrieve `git_diff_applier`
- **Synthesis** (10): "given this design doc, list inconsistencies with the code" — needs new node, tests synthesis

This benchmark gives us a graceful curve of expected library hit rate
(100% → ~40% → 0% across question types). If retrieval helps, we'll see
it most on the first 10 and least on the last 10.

---

## 7. Risks (and what we'd do)

| Risk | Likelihood | Mitigation / pivot |
|---|---|---|
| Base 27B too weak even with library | Medium | Phase 1 A/Bs MIT planner; if needed, drop to using MIT planner exclusively |
| Library retrieval hurts more than helps (planner gets confused by candidates) | Medium | Make retrieval optional; A/B with-library vs without on every benchmark |
| Synthesis success rate too low to compound | Medium | Phase 2 kill criterion catches this; pivot to curated-only library |
| Docker sandbox is slow → verifier becomes bottleneck | Low-Med | Pre-warmed pool of sandbox containers; 30s/verify ceiling |
| Library bloat: catalog grows uselessly | High if no curation | Stitch + LRU + dedup from Phase 2 day 1 |
| LangGraph API churn breaks library | Low | Pin LangGraph in pyproject.toml; versioned catalog with migration scripts |
| LM-generated code in library does something dangerous | Critical if mishandled | Per-node Docker sandbox at verify; non-negotiable safety lint at commit |
| Three benchmarks aren't enough | Low | Add fourth (e.g. SWE-bench-lite) only if first three are conclusive |
| LILO authors release something better while we're building | Low-Med | Stay informed; cite their work; the productionization gap remains regardless |
| ROMA + GEPA already does 90% of this and we're duplicating | Med | Phase 1 includes ROMA-as-planner A/B; if ROMA wins outright, fork ROMA and add only the library layer |
| Trajectory data poisoning (training on bad traces teaches bad behavior) | High if rushed | Don't SFT on our own trajectories until we have leaf_calls > 0 + verified-correct samples |
| protoLabs/fast leaks `<think>` again after some upgrade | Low | Reasoning-parser fix is in CLAUDE.md; covered by config tests |

---

## 8. Repo extraction strategy

**Decision: extract to `~/dev/recurse` (placeholder name) as a standalone
repo for publication and external reproduction.** Lab monorepo keeps a thin
`experiments/rlm/` shim with a README pointing at the new repo.

### Name candidates (final pick TBD)

| Name | Pro | Con |
|---|---|---|
| `recurse` | Short, evocative | Generic, possibly taken |
| `compound-rlm` | Names the differentiator | Awkward |
| `proto-rlm` | Brand fit | Sounds derivative |
| `nodelab` | Names the artifact | Doesn't convey RLM lineage |
| `rlm-library` | Descriptive | Long, boring |

**Default to `compound-rlm` unless better surfaces.** Captures the
differentiation (compounding skill across queries) and isn't squatted on PyPI.

### Repo layout (target)

```
compound-rlm/
├── README.md                 # quickstart, architecture diagram, citation BibTeX
├── PROPOSAL.md               # this document
├── CHANGELOG.md              # per-version log; semver
├── BENCHMARKS.md             # auto-updated per CI run; current numbers
├── LICENSE                   # Apache-2.0 (matches RLM, enables HF publishing)
├── CITATION.cff              # for academic citation
├── pyproject.toml            # uv-based; pinned LangGraph, DSPy, vLLM clients
├── compound_rlm/
│   ├── __init__.py
│   ├── core/                 # the M0 scaffold, refactored
│   │   ├── graph.py
│   │   ├── sandbox.py
│   │   ├── parser.py
│   │   ├── llm.py
│   │   └── prompts.py
│   ├── library/              # Phase 1+
│   │   ├── catalog.py        # the JSONL catalog API
│   │   ├── retrieve.py       # FAISS + type-match
│   │   ├── synthesize.py     # DSPy module
│   │   ├── verify.py         # Docker sandbox runner
│   │   ├── curate.py         # Stitch + LRU + dedup
│   │   └── publish.py        # HF dataset upload
│   ├── nodes/                # canonical hand-built nodes (the seed library)
│   │   ├── git_diff_applier/
│   │   ├── multi_doc_aggregator/
│   │   ├── codebase_grep/
│   │   ├── regex_extract/
│   │   └── chunk_summarize/
│   ├── primitives/           # synthesizer's vocabulary
│   │   ├── exec.py
│   │   ├── leaf_call.py
│   │   ├── partition.py
│   │   ├── aggregate.py
│   │   └── ...
│   ├── dspy_modules/         # DSPy signatures + GEPA-tuned prompts
│   ├── trajectory/           # schema + persistence
│   └── eval/                 # benchmark runners
├── benchmarks/
│   ├── locodiff/             # task loaders + scorers
│   ├── lab_repo_audit/       # 50 questions + ground truth
│   └── longbench_v2/         # subset loader + scorer
├── libraries/                # versioned snapshots (gitattribute LFS)
│   ├── coding-v0.1.0.tar.gz
│   ├── research-v0.1.0.tar.gz
│   └── ...
├── trajectories/             # archived trajectories (LFS)
│   ├── 2026-W18.jsonl
│   └── ...
├── notebooks/                # exploration + paper figures
├── docs/                     # mkdocs site, published to GH Pages
│   ├── index.md
│   ├── architecture.md
│   ├── benchmarks.md
│   ├── library-format.md
│   └── reproducing.md
├── tests/                    # pytest, current 27 tests + new
├── scripts/
│   ├── reproduce_phase1.sh   # one-command reproduction
│   ├── publish_library.sh    # upload to HF
│   └── ci_benchmark.sh
├── .github/
│   └── workflows/
│       ├── tests.yml
│       ├── benchmarks.yml    # weekly cron, archives to trajectories/
│       └── publish.yml
└── paper/                    # LaTeX source for the workshop submission
    ├── main.tex
    ├── figures/
    └── refs.bib
```

### Reproducibility commitments

The repo holds itself to a higher standard than typical research code:

1. **Locked benchmarks.** Each benchmark loader is versioned; if we update,
   bump the version and report numbers under both.
2. **Pinned dependencies.** `uv.lock` committed; `pyproject.toml` pins
   LangGraph, DSPy, vLLM-client minor versions.
3. **`reproduce_phase1.sh`** runs the full Phase 1 numbers from a clean clone,
   given access to the gateway endpoint.
4. **Trajectories archived** weekly to `trajectories/YYYY-WNN.jsonl`
   (Git LFS). Anyone can replay the exact runs.
5. **Library snapshots** under `libraries/` track every published version;
   diffable across versions with a custom JSONL diff tool.
6. **CI runs benchmarks weekly** with the current `main`; auto-PRs
   `BENCHMARKS.md` updates.
7. **Citation file** (`CITATION.cff`) lets others cite us correctly.

### Publication targets

| Venue | Cadence | Format |
|---|---|---|
| protolabs.studio blog | Per phase exit | "We extended RLM with a learnable library — here's what we found" |
| `protoLabsAI/rlm-library-coding` | Per library snapshot | HF dataset with versioning, README, benchmark scores |
| Workshop paper (NeurIPS / ICLR ML4Code or similar) | Q3 2026 if Phase 2 succeeds | Short paper, code + libraries as supplementary |
| Public repo `protoLabsAI/compound-rlm` | Day 1 of repo extraction | MIT or Apache-2.0 |

---

## 9. Brand fit and positioning

This is where we are honest about why we're building it:

### Why protoLabs (and not, say, a Sonnet wrapper)

We serve organizations that **can't or won't send their data to OpenAI / Anthropic**.
Compliance, IP, sovereignty. Our entire stack — gateway, vLLM, Langfuse,
this lab — is built for that customer. A library-learning RLM stack on
local Qwen makes us competitive with cloud frontier models on long-context
tasks while keeping data local. This is a defensible moat that "fork the
OpenAI Responses API" can't match.

### Why open-source

Two reasons:
1. **The library is the artifact.** A closed library is dead weight; an open one is
   a community knowledge base. The MIT/RLM precedent shows the field rewards openness.
2. **We're a small lab.** Open source is our marketing. Every star, every fork,
   every cited paper builds protoLabs.studio.

### Why publish per-domain libraries

The biggest selling-point: "**bring your own library**." A finance-sector
customer can fork `protoLabsAI/rlm-library-finance`, extend it on their
internal data, keep extensions private, contribute generic improvements
back. That's an ecosystem play, not a product.

---

## 10. Open questions (to revisit at each phase exit)

1. **Type signatures over Python objects** — does Pydantic-on-strings-and-lists
   cover enough to make type-matched retrieval useful, or do we need richer
   typing (Effect-style, gradual typing)?
2. **Synthesis primitives — fixed vs growing vocabulary?** If LM can ALSO add
   new primitives, do we lose the safety/auditability benefit?
3. **Verifier scoring** — for tasks without binary success (open-ended QA),
   how do we score synthesized nodes? LLM-as-judge? Held-out task variants?
4. **Cross-organization library sharing** — what's the trust model for a
   user pulling a library from `someotherorg/rlm-library-mystery`?
   Signed nodes? Per-node auditing UI?
5. **Curriculum learning** — Voyager auto-generated its curriculum. Should
   we, or do real user queries provide enough signal?
6. **When to stop synthesizing and just train a planner** — Phase 2 produces
   bounded benefits; at some point post-training is the right move. What's
   the trigger?
7. **Multi-tenant catalog** — if this becomes a product, multiple
   organizations share the runtime but have private libraries. Architecture?
8. **Library quality vs. user choice** — should the planner ALWAYS pick the
   highest-success-rate node, or sometimes explore? Bandit-style?

---

## 11. Decision log (will grow)

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-02 | Build standalone repo `compound-rlm`, separate from lab monorepo | Publication discipline; reproducibility; ecosystem ambition |
| 2026-05-02 | Use DSPy + LangGraph as substrate; don't reinvent | Stop building infrastructure that already exists |
| 2026-05-02 | Three benchmarks (LoCoDiff Q3-5, lab-repo audit, LongBench v2) | Stress hypothesis from different angles |
| 2026-05-02 | Phase 1 includes A/B with MIT's `rlm-qwen3-8b-v0.1` as planner | Disentangle "library helps" from "MIT planner helps" |
| 2026-05-02 | Verifier in Docker sandbox, not local exec | Safety non-negotiable for LM-generated code commit |
| 2026-05-02 | Library compaction via Stitch (or principled equivalent) from Phase 2 day 1 | Catalog bloat is a known failure mode |

---

## 12. Immediate next steps (after iter-3 finishes)

1. Update `EXPERIMENTS.md` with iter-3 results (final M0 baseline numbers).
2. Create `~/dev/compound-rlm` repo via `git init`. Migrate `experiments/rlm/`
   contents. Set up pyproject + uv lock.
3. Build the lab-repo audit benchmark: 50 hand-written questions, expected
   answers stored as JSONL.
4. Download `mit-oasys/rlm-qwen3-8b-v0.1` to `/mnt/models/`. Add a vLLM systemd
   unit on a third port (e.g. 8003). Test as alternative planner.
5. Begin Phase 1 build: trajectory annotation, DSPy module wrapping,
   first hand-built node (`git_diff_applier`).

Tasks 1-5 above are the unblock. Once they're done, Phase 1 work proper begins.

---

*End of proposal. Critique welcome.*
