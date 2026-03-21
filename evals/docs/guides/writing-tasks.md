# Writing Eval Tasks

## Principles (from Anthropic)

1. **Start with real failures** — convert bugs and user complaints into test cases
2. **Make tasks unambiguous** — two experts should agree on pass/fail independently
3. **Test both sides** — if testing "agent should search", also test "agent should NOT search"
4. **Isolate trials** — no shared state between runs (clean environment each time)
5. **Grade outcomes, not paths** — check what was produced, not how

## Step-by-Step

### 1. Identify what to test

Good sources for eval tasks:
- Bugs you've fixed in protoClaw
- Langfuse traces with low scores
- Edge cases you've seen in production
- Capabilities you're building (browser, coding, tool use)

### 2. Write the task YAML

```yaml
id: coding_001
name: "Fix Off-by-One Bug"
difficulty: medium
category: coding

system_prompt: |
  You are a coding assistant. Fix bugs in the provided code.

prompt: |
  This function should return the sum of numbers 1 to n (inclusive),
  but it has a bug. Fix it:

  ```python
  def sum_to_n(n):
      total = 0
      for i in range(n):
          total += i
      return total
  ```

expected:
  contains: "range(1, n + 1)"
  # or: "range(n + 1)" starting from 0

graders:
  - type: llm_judge
    dimension: correctness
    rubric: |
      The fix should change `range(n)` to `range(1, n + 1)` or equivalent.
      The function should return n*(n+1)/2 for any positive n.
      1.0 = correct fix
      0.5 = partially correct (e.g., fixes loop but introduces another issue)
      0.0 = wrong fix or no fix
  - type: llm_judge
    dimension: explanation
    rubric: |
      Did the agent explain WHY the bug exists (off-by-one, range excludes end)?
      1.0 = clear explanation of the root cause
      0.5 = vague explanation
      0.0 = no explanation
```

### 3. Add mock tool responses (if applicable)

If the task involves tools, provide deterministic mock responses:

```yaml
tools:
  - type: function
    function:
      name: read_file
      description: "Read a file from the filesystem"
      parameters:
        type: object
        properties:
          path:
            type: string
        required: ["path"]

mock_tool_responses:
  read_file: |
    {"content": "def sum_to_n(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"}
```

### 4. Run and calibrate

```bash
# Test with a strong model first
python -m runners.run_custom \
    --task tasks/coding/fix_off_by_one.yaml \
    --model claude-sonnet-4-6

# If pass@10 is 0%, the task is probably broken
# If pass@1 is 100%, the task is too easy
```

### 5. Organize by category

```
tasks/
├── tool_use/           # Tool selection and invocation
├── browser/            # Web browsing tasks
├── coding/             # Code gen, bug fixes, refactoring
└── safety/             # Prompt injection, boundary testing
```

## Anti-Patterns

- **Path-checking**: Don't assert specific tool call sequences. Agents find valid alternatives.
- **One-sided tests**: Don't only test positive cases. Test when the agent should decline or ask for clarification.
- **Shared state**: Don't let trials read state from previous trials. Each trial must start clean.
- **Ambiguous grading**: If you can't tell whether an output passes, the task needs clearer criteria.
- **Synthetic-only**: Tasks from real failures are 10x more valuable than contrived puzzles.
