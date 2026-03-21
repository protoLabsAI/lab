# Infrastructure Overview

## System Diagram

```
                    Internet
                       │
                Cloudflare Tunnel
                       │
         ┌─────────────┼─────────────┐
         v             v             v
    ai.proto-labs.ai  chat.proto-labs.ai  traces.proto-labs.ai
    Gateway :4000     Open WebUI :3000    Langfuse :3001
         │
         ├── vLLM :8000 (local models)
         ├── Claude, GPT, Gemini (cloud APIs)
         ├── OpenRouter (DeepSeek, GLM, Kimi, etc.)
         └── Langfuse auto-tracing
```

## Nodes

| Node | Role | Hardware |
|------|------|----------|
| **protolabs** (this machine) | AI inference, experiments, training | 2x RTX PRO 6000 Blackwell (192GB VRAM) |
| **pve01** | Control plane (Infisical, Prometheus, Grafana) | — |

Nodes connected via Tailscale MagicDNS.

## Services on protolabs

| Service | Port | Runtime | Purpose |
|---------|------|---------|---------|
| vLLM | :8000 | systemd | LLM inference (model swapped via `vllm-swap.sh`) |
| Gateway | :4000 | Docker | LiteLLM proxy — unified API for all providers |
| Langfuse | :3001 | Docker | LLM observability (traces, scores, experiments) |
| Open WebUI | :3000 | Docker | Chat UI |
| ComfyUI | :8188 | systemd | Image/video generation |
| protoClaw | :7865 | Docker | Sandboxed AI agent |
| node-exporter | :9100 | Docker | Prometheus node metrics |
| cAdvisor | :8080 | Docker | Container metrics |
| GPU exporter | :9835 | Docker | NVIDIA GPU metrics |

## Secrets

All secrets in Infisical at `secrets.proto-labs.ai` (hosted on pve01).

| Project | Consumers |
|---------|-----------|
| secret-management | Gateway, Langfuse, protoClaw |

Machine Identity (Universal Auth) for automated injection via `start.sh` wrappers.
