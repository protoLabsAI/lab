# Reference: Claw-Eval

## Overview

[Claw-Eval](https://github.com/claw-eval/claw-eval) is a 104-task benchmark for
evaluating LLMs as autonomous agents in personal assistant scenarios. Integrated
as a git submodule at `claw/claw-eval/`.

## Key Concepts

- **Pass^3**: Task passes only if agent succeeds in ALL 3 independent trials
- **Mock Services**: 15 FastAPI services (Gmail, Calendar, CRM, etc.) with deterministic data
- **Docker Sandbox**: Isolated execution with shell, file I/O, and browser tools
- **Multi-dimensional scoring**: 80% completion + 20% robustness, with safety multiplier

## Task Categories

| Category | Tasks | Examples |
|----------|-------|---------|
| Communication | T01-T06 | Email triage, reply drafting |
| Scheduling | T03-T04, T29-T30 | Calendar management |
| Productivity | T07-T14 | Todo, expense reports, notes |
| Operations | T17-T24 | Ticket triage, CRM, dashboards |
| Research | T43-T50 | CVE research, vendor comparison |
| Multimodal | T51-T58 | Image ID, document QA |
| Code/Technical | T68-T70, T100-T104 | Schema migration, packet decoder |
| Safety | T73-T75 | Prompt injection defense |

## Scoring Formula

```
base = 0.80 * completion + 0.20 * robustness
final = base * safety_factor
passed = final >= 0.75
```

## Usage with Our Gateway

```bash
# Config points at gateway
cat claw/config.yaml
# model.base_url: http://localhost:4000/v1

# Run tasks
python -m runners.run_claw --model local --tasks T01,T02
python -m runners.run_claw --model claude-sonnet-4-6 --all-tasks
```

## CLI Reference

```bash
claw-eval run <task_id> [--trials N] [--config path]
claw-eval batch [--trials N] [--workers N] [--continue]
claw-eval grade [--config path]      # re-grade without re-running
claw-eval list                       # show available tasks
claw-eval build-image                # build Docker sandbox
claw-eval cleanup                    # remove leftover containers
```
