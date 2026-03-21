# protoLabs AI Gateway

Unified AI inference gateway + observability for the protoLabs homelab. Routes all LLM requests through a single OpenAI-compatible endpoint, with full tracing via Langfuse and secrets managed by Infisical.

## Architecture

```
  Your apps (protoClaw, experiments, agents, etc.)
                    │
          ┌─────────▼──────────┐
          │  AI Gateway :4000  │  ← single OpenAI-compatible endpoint
          │  LiteLLM Proxy     │
          └──┬───┬───┬───┬────┘
             │   │   │   │
  ┌──────┐ ┌─┴─┐ │  ┌┴────────────┐
  │ vLLM │ │Oll│ │  │ Cloud APIs   │
  │:8000 │ │ama│ │  │ OpenAI       │
  └──────┘ └───┘ │  │ Anthropic    │
                 │  │ Gemini, Groq │
          ┌──────┘  │ Grok, OR     │
          │         └──────────────┘
    ┌─────▼──────┐
    │ Langfuse   │  ← tracing, evals, cost tracking
    │ :3001      │
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │ Infisical  │  ← secret management (hosted on pve01)
    │ secrets.   │
    │ proto-     │
    │ labs.ai    │
    └────────────┘
```

## Stack overview

| Service | Port | Description |
|---------|------|-------------|
| **LiteLLM Gateway** | `:4000` | OpenAI-compatible API proxy — routes to local + cloud LLMs |
| **LiteLLM Admin UI** | `:4000/ui` | Key management, usage dashboard, model testing |
| **Langfuse** | `:3001` | LLM observability — traces, evals, cost tracking, prompt management |
| **Langfuse Worker** | internal | Async event processor (ClickHouse ingestion) |
| **ClickHouse** | internal | Analytics/trace storage (OLAP) |
| **Redis** | internal | Queue + cache for async ingestion |
| **MinIO** | `:9095` | S3-compatible object storage for trace payloads |
| **PostgreSQL (gateway)** | internal | LiteLLM usage tracking + key management |
| **PostgreSQL (langfuse)** | internal | Langfuse metadata + project config |

## Supported providers

| Provider | Models | Auth |
|----------|--------|------|
| **vLLM** (local) | Any loaded model | No key needed |
| **Ollama** (local) | Any pulled model | No key needed |
| **Anthropic** | Claude Opus/Sonnet/Haiku | `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-4o, o3 | `OPENAI_API_KEY` |
| **Google** | Gemini 2.5 Pro/Flash | `GEMINI_API_KEY` |
| **Groq** | Llama 3.3 70B, 8B | `GROQ_API_KEY` |
| **xAI** | Grok 3, 3 Mini | `XAI_API_KEY` |
| **OpenRouter** | 200+ models | `OPENROUTER_API_KEY` |

## Quick start

```bash
# 1. Clone
git clone https://github.com/protoLabsAI/gateway.git
cd gateway

# 2. Configure secrets (choose one)

# Option A: Infisical (production — no keys on disk)
./start.sh --setup    # one-time: enter machine identity credentials
./start.sh            # fetches secrets from Infisical, starts everything

# Option B: Local .env (dev only)
cp .env.example .env  # edit with your keys
./start.sh --local

# 3. Services are now available:
#    Gateway:  http://localhost:4000/v1
#    Admin:    http://localhost:4000/ui
#    Langfuse: http://localhost:3001

# 4. Stop
./start.sh --stop
```

## Usage from apps

Point any OpenAI-compatible client at the gateway:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="<your LITELLM_MASTER_KEY>",
)

# Local vLLM model
response = client.chat.completions.create(
    model="vllm/Qwen3.5-27B",
    messages=[{"role": "user", "content": "hello"}],
)

# Cloud model — same endpoint, just change the model name
response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "hello"}],
)
```

All requests are automatically traced to Langfuse.

## Model routing

| Request model | Routes to |
|---------------|-----------|
| `vllm/*` | Local vLLM on :8000 |
| `ollama/*` | Local Ollama on :11434 |
| `claude-*` | Anthropic API |
| `gpt-*`, `o3` | OpenAI API |
| `gemini-*` | Google AI API |
| `groq-*` | Groq API |
| `grok-*` | xAI API |
| `openrouter/*` | OpenRouter API |

## Langfuse (observability)

Self-hosted [Langfuse v3](https://langfuse.com) running at `:3001`. Every request through the gateway is automatically traced with:

- Model, provider, latency, status
- Token counts and cost
- Full request/response content
- Reasoning/thinking tokens (when available)

### Setup

1. Open `http://<host>:3001` and create an account
2. Create a project (e.g., "gateway")
3. Go to **Settings** > **API Keys** and create a key pair
4. Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to Infisical
5. Re-run `./start.sh` to pick up the new keys

### Langfuse stack components

| Component | Image | Purpose |
|-----------|-------|---------|
| langfuse-web | `langfuse/langfuse:3` | UI + API |
| langfuse-worker | `langfuse/langfuse-worker:3` | Async event processor |
| langfuse-clickhouse | `clickhouse/clickhouse-server` | Trace/analytics storage |
| langfuse-redis | `redis:7-alpine` | Ingestion queue + cache |
| langfuse-minio | `minio/minio` | Object storage for payloads |
| langfuse-db | `postgres:16-alpine` | Project metadata (UTC enforced) |

## Secrets management (Infisical)

All secrets are stored in [Infisical](https://secrets.proto-labs.ai) — no API keys on disk in production.

### How it works

1. `start.sh` authenticates with Infisical via Universal Auth (Machine Identity)
2. Fetches all secrets for the gateway project
3. Injects them as environment variables into `docker compose up`
4. Docker Compose substitutes `${VAR}` placeholders in the compose file

Machine identity credentials live at `~/.config/gateway/identity` (mode 600, gitignored).

### Infisical project: `secret-management`

**URL:** `https://secrets.proto-labs.ai`

**Required secrets:**

| Secret | Description | How to generate |
|--------|-------------|-----------------|
| `LITELLM_MASTER_KEY` | Gateway API auth key | `sk-gateway-$(openssl rand -hex 16)` |
| `GATEWAY_DB_PASSWORD` | LiteLLM PostgreSQL password | `openssl rand -hex 16` |
| `LANGFUSE_NEXTAUTH_SECRET` | Langfuse session signing | `openssl rand -hex 32` |
| `LANGFUSE_SALT` | Langfuse API key hashing | `openssl rand -hex 32` |
| `LANGFUSE_ENCRYPTION_KEY` | Langfuse data encryption (64 hex chars) | `openssl rand -hex 32` |
| `LANGFUSE_DB_PASSWORD` | Langfuse PostgreSQL password | `openssl rand -hex 16` |
| `LANGFUSE_CLICKHOUSE_PASSWORD` | ClickHouse password | `openssl rand -hex 16` |
| `LANGFUSE_REDIS_PASSWORD` | Redis password | `openssl rand -hex 16` |
| `LANGFUSE_MINIO_USER` | MinIO access key | `langfuse` |
| `LANGFUSE_MINIO_PASSWORD` | MinIO secret key | `openssl rand -hex 16` |
| `LANGFUSE_URL` | Public Langfuse URL | `http://<host>:3001` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse project public key | From Langfuse UI |
| `LANGFUSE_SECRET_KEY` | Langfuse project secret key | From Langfuse UI |

**Cloud provider keys (add as needed):**

| Secret | Provider |
|--------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google AI |
| `GROQ_API_KEY` | Groq |
| `XAI_API_KEY` | xAI (Grok) |
| `OPENROUTER_API_KEY` | OpenRouter |
| `VLLM_API_KEY` | vLLM (set to `none`) |

### First-time Infisical setup

1. Go to `https://secrets.proto-labs.ai`
2. Create or open the gateway project
3. Add all required secrets (use "Paste Secret Values" with `.env` format)
4. **Organization Settings** > **Machine Identities** > create `gateway-ava-ai` (Admin role)
5. Enable **Universal Auth** on the identity, copy Client ID + Client Secret
6. On ava-ai: `./start.sh --setup` — enter Client ID, Client Secret, and Project ID

### Important notes

- All passwords must be **hex-only** (`openssl rand -hex`) — base64 characters (`+`, `/`, `=`) break PostgreSQL connection URLs
- Langfuse PostgreSQL **must run UTC** — this is set in the compose file via `TZ=UTC` and `PGTZ=UTC`
- `LANGFUSE_ENCRYPTION_KEY` must be exactly 64 hex characters
- After changing secrets in Infisical, re-run `./start.sh` to pick up changes

## Configuration files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Full stack definition (gateway + langfuse + supporting services) |
| `config.yaml` | LiteLLM model routing, callbacks, settings |
| `start.sh` | Secure startup script with Infisical integration |
| `.env.example` | Template for local dev (copy to `.env`) |

## License

MIT
