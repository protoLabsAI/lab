# User Access & External Exposure

Guide for adding users to the gateway and exposing services through Cloudflare.

## API Key Management

LiteLLM has built-in per-user API key management with budgets, rate limits, and usage tracking.

### Create a user key

```bash
# Via API
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer <LITELLM_MASTER_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "key_alias": "josh-personal",
    "max_budget": 50,
    "duration": "30d",
    "models": ["local", "claude-sonnet-4-6", "gemini-2.5-flash"],
    "metadata": {"user": "josh"}
  }'
```

Or use the admin UI at `http://<host>:4000/ui` > Keys > Create Key.

### Key options

| Option | Description | Example |
|--------|-------------|---------|
| `key_alias` | Human-readable name | `"josh-mobile"` |
| `max_budget` | Max spend in USD (cloud models only) | `50` |
| `duration` | Key expiry | `"30d"`, `"7d"`, `"365d"` |
| `models` | Restrict to specific models | `["local", "claude-sonnet-4-6"]` |
| `tpm_limit` | Tokens per minute | `100000` |
| `rpm_limit` | Requests per minute | `60` |
| `max_parallel_requests` | Concurrent request limit | `10` |

### List / revoke keys

```bash
# List all keys
curl http://localhost:4000/key/info \
  -H "Authorization: Bearer <LITELLM_MASTER_KEY>"

# Delete a key
curl -X POST http://localhost:4000/key/delete \
  -H "Authorization: Bearer <LITELLM_MASTER_KEY>" \
  -d '{"keys": ["sk-abc123..."]}'
```

### Key hierarchy

```
Master Key (LITELLM_MASTER_KEY)
  └── Full admin access, key management, config changes
  └── NEVER share externally

Virtual Keys (per-user)
  └── Scoped to specific models
  └── Budget-limited (cloud spend capped)
  └── Rate-limited (RPM/TPM)
  └── Expiring (duration-based)
  └── Safe to distribute
```

## Cloudflare Tunnel Exposure

### Tunnel routes to add

Add these to your Cloudflare tunnel config (managed in `homelab-iac`):

| Hostname | Service | Purpose | Auth |
|----------|---------|---------|------|
| `ai.proto-labs.ai` | `http://localhost:4000` | Gateway API | API key (per-user virtual keys) |
| `traces.proto-labs.ai` | `http://localhost:3001` | Langfuse UI | Email/password (Langfuse built-in) |
| `chat.proto-labs.ai` | `http://localhost:3000` | Open WebUI | Email/password (WebUI built-in) |

### Cloudflare Access (recommended)

For additional security, add Cloudflare Access policies:

- **Gateway API** (`ai.proto-labs.ai`) — bypass Access (API key auth is sufficient), or add Access for browser-based admin UI at `/ui`
- **Langfuse** (`traces.proto-labs.ai`) — require Cloudflare Access (email allowlist or SSO)
- **Open WebUI** (`chat.proto-labs.ai`) — require Cloudflare Access or rely on built-in auth

### Example tunnel config

```yaml
tunnel: <tunnel-id>
credentials-file: /etc/cloudflare/<tunnel-id>.json

ingress:
  - hostname: ai.proto-labs.ai
    service: http://localhost:4000
  - hostname: traces.proto-labs.ai
    service: http://localhost:3001
  - hostname: chat.proto-labs.ai
    service: http://localhost:3000
  - service: http_status:404
```

## User onboarding checklist

1. Create a virtual key with appropriate budget and model access
2. Share the key and endpoint: `https://ai.proto-labs.ai/v1`
3. User configures their client:
   ```python
   client = OpenAI(
       base_url="https://ai.proto-labs.ai/v1",
       api_key="sk-<their-virtual-key>",
   )
   ```
4. Optionally invite them to Langfuse for trace visibility
5. Optionally create Open WebUI account for chat access

## Cost control

- Local models (`local`, `ollama/*`) cost $0 — no budget impact
- Cloud models (Claude, GPT, Gemini, etc.) consume budget based on token pricing
- Set `max_budget` per key to cap cloud spend
- Monitor usage in admin UI at `/ui` or Langfuse
- Keys auto-disable when budget is exhausted (user gets 429 error)

## Security considerations

- Master key must stay in Infisical — never expose via Cloudflare
- Virtual keys are safe to distribute (scoped, budget-limited, expiring)
- Cloudflare Access adds network-level auth on top of application auth
- All traffic through Cloudflare tunnel is encrypted (no direct port exposure)
- Gateway admin UI (`/ui`) should be protected by Cloudflare Access if exposed
- Langfuse stores full request/response content — restrict access to trusted users
