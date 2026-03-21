# Task Format

Custom eval tasks are defined as YAML files in `tasks/`.

## Schema

```yaml
# Required
id: tool_use_001                    # Unique identifier
name: "Basic Calculator"            # Human-readable name
prompt: |                           # The prompt sent to the agent
  What is 847 * 293 + 1456?

# Optional
difficulty: simple | medium | hard  # For categorization
category: tool_use                  # Task category (matches directory)

system_prompt: |                    # System message for the agent
  You are a helpful assistant.

tools:                              # OpenAI-format tool definitions
  - type: function
    function:
      name: calculator
      description: "Perform arithmetic"
      parameters:
        type: object
        properties:
          expression:
            type: string
        required: ["expression"]

mock_tool_responses:                # Simulated tool outputs
  calculator: '{"result": 249527}'

expected:                           # Ground truth for grading
  answer: "249527"
  tool_used: true

graders:                            # Grading criteria
  - type: llm_judge
    dimension: correctness
    rubric: |
      Score whether the answer is correct...
  - type: llm_judge
    dimension: tool_usage
    rubric: |
      Score whether the agent used tools...
```

## Task Categories

| Directory | Tests | Examples |
|-----------|-------|----------|
| `tool_use/` | Correct tool selection and invocation | Calculator, file ops, API calls |
| `browser/` | Web browsing and information extraction | Page reading, form filling, navigation |
| `coding/` | Code generation and execution | Bug fixes, implementations, refactoring |
| `safety/` | Prompt injection defense, guardrails | Injection attempts, boundary testing |

## Writing Good Tasks

1. **Unambiguous** — Two experts should independently reach the same pass/fail verdict
2. **Balanced** — Test when behavior should AND should not occur
3. **Reproducible** — Mock external dependencies, no shared state between trials
4. **Partial credit** — Multiple grading dimensions with weighted scoring
5. **Real-world** — Derived from actual agent failures, not synthetic puzzles

If pass@100 is 0%, the task is probably broken, not the agent.
