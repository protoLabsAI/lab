# Agentic-data dataset census (2026-07-06)

Completes the open-dataset census `RESEARCH.md` deferred ("sanity-check any committed play
against HF dataset search first"). Goal: source a "bigger + better" training set to distill
**Ornith-1.0-35B → Qwen3.5-2B** (see `evals/baselines/README.md` bake-off + [[project_distill_base_decision]]).
Four parallel researchers, every dataset grounded on its live HF card / arXiv / GitHub this session.

## The strategy the census collapses to (two moves)

1. **Train directly** on the ~6 confirmed license-clean agentic corpora below (fast breadth, ~250k rows).
2. **Don't ingest tokens for anything in-domain + verified — reuse the ENVIRONMENT and regenerate
   with Ornith.** The best claw-domain signals (τ-bench retail/airline/telecom) are *environments,
   not datasets*; rolling Ornith through them and reward-filtering gives clean-license,
   deterministically-verified, in-domain trajectories with zero third-party redistribution risk.
   Same move neutralizes the GPT-4-tainted sets (APIGen-MT, ToolBench): reuse their task
   specs / tool schemas / DB envs as scaffolds, train on Ornith's tokens.

Provenance axis that decides "clean": **DeepSeek-R1 (MIT) and Qwen/QwQ (Apache) outputs carry no
anti-compete clause → clean. GPT-4/Claude outputs are ToS-tainted for a published commercial model
regardless of the dataset's license tag. Llama-generated is commercial-OK but drags the Llama
Community License (mandatory "Llama-" naming, AUP, <700M-MAU) onto the student.**

## Tier A — train directly (license-clean, confirmed)

    dataset                                     license      rows    provenance        role
    Team-ACE/ToolACE                            apache-2.0   11.3k   non-OpenAI synth  anchor multi-turn tool-use
    interstellarninja/hermes_reasoning_tool_use apache-2.0   51k     DeepHermes-3(open) best format fit (<think>+<tool_call>)
    HuggingFaceTB/smoltalk2                     apache-2.0   ~3.4M   Qwen3/DeepSeek-V3 best clean+agentic (smolagents FC)
    nvidia/When2Call                            cc-by-4.0    27.6k   synth (commercial-OK) teaches when-NOT-to-call + det. MCQ eval
    AgentGym/AgentTraj-L                        MIT repo*    14.5k   env-verified      best non-SWE trajectory breadth (14 envs)
    OS-Copilot/OS-Genesis-mobile-data           MIT          51k     rule-verified     claw ops (cal/contacts/SMS/tasks); use a11y-tree text
    HuggingFaceTB/smoltalk                      apache-2.0   1.1M    Llama-3.1-405B    FC/multi-turn (Llama-license caveat)
    glaiveai/glaive-function-calling-v2         apache-2.0   113k    undisclosed       volume filler (provenance flag)

    reasoning fuel (harden the student underneath the agentic data, all clean):
    open-thoughts/OpenThoughts3-1.2M            apache-2.0   1.2M    QwQ-32B
    PrimeIntellect/SYNTHETIC-1-SFT-Data         apache-2.0   894k    DeepSeek-R1
    nvidia/Llama-Nemotron-Post-Training         cc-by-4.0    ~3.9M   mixed → filter to DeepSeek/Qwen subsets only

    * AgentTraj-L: repo MIT but dataset-artifact license field is missing — CONFIRM before publish.

## Tier B — environments to regenerate through Ornith (clean tokens, in-domain, deterministic)

    τ-bench / τ²-bench   sierra-research/tau-bench   MIT env   retail/airline/telecom, deterministic DB-state reward  ← FLAGSHIP claw-domain path
    AgentGym envs        WooooDyy/AgentGym-RL        MIT       WebShop/ALFWorld/SciWorld/BIRD-SQL/BabyAI, vLLM-native
    ASTRA env_synthesis  LianjiaTech/astra           apache    QA→code-executable rule-verifiable RLVR envs
    (scaffold-only, don't ingest tokens): APIGen-MT-5k task specs, ToolBench RapidAPI schemas

## Tier C — synthesis frameworks to stand up (generate our own at scale)

    ASTRA          LianjiaTech/astra        apache-2.0  local-friendly  DETERMINISTIC verifier  ← TOP: matches our code-exec grader stack
    AgentGym-RL    WooooDyy/AgentGym-RL     MIT         vLLM-NATIVE     env-driven reward        ← lowest-friction, GRPO/PPO wired
    TextArena      TextArena/TextArena      MIT         local           deterministic (games)   games-only gym
    OpenHands      OpenHands/OpenHands      MIT         local (LiteLLM) test-verified            SWE-only (saturated — skip)

    AgentWorld (ours): env simulator. NOT for the verified shard — ASTRA/AgentGym ship REAL
    deterministic verifiers; AgentWorld can't verify its own state ("truth about process, lies
    about values"). Its niche = breadth / PROCESS-imitation environments for the SFT shard, gated
    on the AgentWorld fidelity benchmark (RESEARCH.md play #2). Run the fidelity probe before
    training on any AgentWorld-generated trajectory.

## AVOID (license/ToS traps — the "looks clean, isn't" set)

    Salesforce/APIGen-MT-5k        cc-by-NC + GPT-4 + explicit anti-OpenAI-compete   double-blocked (env-template only)
    microsoft/orca-agentinstruct-1M cdla-permissive TAG masking Azure GPT-4 gen      sneakiest: largest agentic set, OpenAI-tainted
    OpenBMB/ToolBench              apache tag but gpt-3.5-generated                  ToS taint (schemas-only)
    allenai/tulu-3-sft-mixture     ODC-BY hides NC (No Robots) + GPT-4 (WildChat)    never train wholesale; rebuild from clean subsets
    Salesforce/xlam-fc-60k         cc-by-4.0 TAG vs "research only" card text        unresolved conflict — clear w/ Salesforce first
    SYNTHAGENT                     no LICENSE file                                    outputs unusable
    CUA-Gym default path           hardcodes --model gpt-4o                          taints provenance unless client swapped

## Corrections to prior notes
- Hack-verifiable-environments paper is **arXiv 2605.20744** (RESEARCH.md finding #7 cited 2606.26300 — wrong).
- Agents-A1 data engine confirmed CLOSED (repo = weights+eval+deploy only). Kimi/DeepSeek synth = no code.
- Watch (paper-only, code may drop): EigenData (2601.22607) — promises τ-bench Airline/Telecom executable checkers.

## Recommended build (immediate)
1. **Tier-A direct-train mix** (ToolACE + hermes_reasoning_tool_use + smoltalk2 + When2Call + AgentTraj-L +
   OS-Genesis-text) — the breadth SFT floor, all clean. τ²-bench stays HELD-OUT eval, never trained on.
2. **Stand up AgentGym-RL first** (MIT, vLLM-native, sized for our regime) then **ASTRA** (deterministic
   verifier) as the two Ornith-rollout generators; τ-bench env as the in-domain claw generator.
3. game-rlvr byte-replay + code-exec = the deterministic-verifier backbone we already own.
