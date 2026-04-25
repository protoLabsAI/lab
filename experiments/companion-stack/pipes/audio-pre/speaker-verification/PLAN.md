# speaker-verification — owner vs stranger

**Pipe**: audio-pre.
**Status**: planned (Phase 1).

## Problem

ORBIS is single-owner. The orb should know when a *different person*
is talking — for any of:

- Refusing to act on commands from non-owners
- Different LLM persona / restricted memory access
- Tagging session metadata ("who was talking when")
- Family-mode (recognize multiple known voices, fall back for unknown)

Today ORBIS treats every audio frame as the owner's. That's a
single-line vulnerability and a missed UX opportunity.

## Why ORBIS needs it specifically

- ORBIS is hosted on tailnet. Anyone on the tailnet (a family
  member, a guest, a phone left unlocked) can talk to the orb.
- The companion layer's *whole point* is owner-personalization. A
  guest shouldn't trigger personality drift.
- Recording a guest's voice into long-term `facts` table is
  borderline a privacy issue.

## Candidate architectures

1. **Off-the-shelf speaker embedding model + cosine similarity gate**
   - `speechbrain/spkrec-ecapa-voxceleb` (~6 M params, 192-dim
     embedding)
   - `pyannote/embedding`
   - Enrollment: 5-10 sec of owner audio at first run; cache embedding.
   - Inference: 16-dim cosine vs cached. Threshold tuning.
   - **Pro**: zero training, ships today.
2. **Fine-tuned ECAPA-TDNN on owner-specific data**
   - 2-3 min of owner audio + VoxCeleb negatives → contrastive fine-tune
   - **Pro**: better separation under noisy ORBIS conditions.
   - **Con**: requires per-owner training; does ORBIS want that?
3. **ContiguousID + speaker embedding for multi-owner mode**
   - Cluster speaker embeddings across sessions, label clusters
     ("Owner", "Family member 1", "Stranger").

## Datasets

- **VoxCeleb 1 + 2** — public, ~7000 speakers, gold standard for SV
  benchmarking. Held-out test set is the standard for EER reporting.
- **Self-recorded** owner enrollment audio (~5 min, varied
  conditions: close, room, background-noise).
- **Mozilla Common Voice** — public negatives (other speakers) for
  contrastive fine-tune if needed.

## Eval plan

1. **EER (Equal Error Rate)** on VoxCeleb1-test as the literature
   benchmark.
2. **Owner-vs-rest** on a held-out user set: 50 owner clips + 50
   non-owner clips, threshold-swept.
3. **Robustness sweep**: clean / noisy / over-the-phone / different
   mic distances. ORBIS will see all of these.
4. **Latency**: < 30 ms on Blackwell, < 200 ms on CPU (for
   on-orbis-host inference).

## Deliverables

- HF model: `protoLabsAI/orbis-speaker-verifier-v0` (likely just an
  ECAPA-TDNN config + inference wrapper around the speechbrain
  weights — research artifact more than novel weights).
- Pipecat frame processor: `voice/agent/speaker_gate.py`.
- ORBIS integration: `config/users.yaml` gets a `voiceprint_path`
  field; non-owner audio gets a different LLM persona or a
  `delegate_to=guest_handler` route.
- Blog post: "Single-owner voice agents need to know who's talking."

## Open questions

- Is enrollment a one-time onboarding step, or does the orb
  continuously refine the owner embedding?
- What's the right UX for false-rejects (orb says "I don't know
  you, are you the owner?")? How is enrollment retriable?
- Does ORBIS want true multi-owner mode (Family) at v1, or just
  binary owner/stranger?

## Dependencies

- None blocking. Can start immediately.
- Should land before audio-tags graduates to ORBIS, since the gate
  is upstream of mood-injection (don't update owner's mood from a
  guest's voice).
