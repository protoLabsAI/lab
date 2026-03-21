# Function Calling Evals

Tests whether models produce correct structured tool calls: right function name,
correct arguments, proper JSON structure, and appropriate restraint (not calling
tools when they shouldn't).

## Running

```bash
# Run basic test suite
./run.sh function-call --model local --suite basic

# Run edge cases
./run.sh function-call --model local --suite edge_cases

# Run all suites
./run.sh function-call --model local --all-suites

# Submit scores to Langfuse
./run.sh function-call --model local --all-suites --submit-langfuse
```

## Scoring

The `FunctionCallGrader` scores each expected tool call on two dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Name accuracy | 50% | Did the model call the right function? |
| Argument accuracy | 50% | Are the arguments correct? (fuzzy matching) |

Extra unexpected calls incur a 5% penalty each (capped at 20%).

## Writing Test Cases

Test cases are YAML files in `function_call/test_cases/`:

```yaml
name: "My Test Suite"
tests:
  - id: test_001
    prompt: "What's the weather in NYC?"
    tools:
      - type: function
        function:
          name: get_weather
          parameters:
            type: object
            properties:
              city: { type: string }
            required: ["city"]
    expected:
      tool_calls:
        - name: get_weather
          arguments:
            city: "New York"
```

### Key scenarios to test

1. **Single tool call** — basic accuracy
2. **Parallel tool calls** — "check weather in NYC and London" → two calls
3. **Tool restraint** — "just say hello" with tools available → no calls
4. **Missing info** — "book a flight" without origin/destination → should ask, not guess
5. **Nested/complex params** — arrays, dates, enums
