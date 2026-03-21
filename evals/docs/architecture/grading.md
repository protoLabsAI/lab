# Grading Framework

## Three Grader Types

### 1. Outcome Graders (deterministic)

Fast, cheap, reproducible. Check the final state, not the process.

```python
from graders.outcome import OutcomeGrader, contains_string, tool_was_called

grader = OutcomeGrader(
    dimension="correctness",
    assertions=[
        contains_string("output", "249527"),
        tool_was_called("calculator"),
    ],
    threshold=0.75,
)
result = grader.grade(task_input, task_output)
# GradeResult(dimension="correctness", score=1.0, passed=True)
```

**When to use:** Verifiable facts, file creation, API call verification, format checks.

**Built-in assertions:**
- `contains_string(key, substr)` — output contains expected text
- `matches_regex(key, pattern)` — output matches pattern
- `equals(key)` — exact match with expected
- `file_exists(path)` — file was created
- `json_valid(key)` — output is valid JSON
- `tool_was_called(name)` — specific tool was invoked

### 2. LLM Judge (model-based)

Flexible, captures nuance. Uses a cloud model (via gateway) to score against a rubric.

```python
from graders.llm_judge import LLMJudge

judge = LLMJudge(
    dimension="helpfulness",
    model="claude-sonnet-4-6",
    rubric="Rate how helpful the response is... {input} {output} {expected}",
)
result = judge.grade(task_input, task_output, expected)
# GradeResult(dimension="helpfulness", score=0.85, passed=True, reasoning="...")
```

**When to use:** Subjective quality, communication style, reasoning depth, creative tasks.

**Best practices:**
- Grade each dimension in isolation (separate judge call per dimension)
- Provide clear rubrics with score anchors (1.0 = excellent, 0.75 = good, etc.)
- Give the judge a way out (0.5 when unsure)
- Use a stronger model as judge (claude-sonnet-4-6 judging local model output)
- Calibrate against human judgment quarterly

### 3. Langfuse Scorer (integration)

Submits grades to Langfuse for storage, comparison, and the eval flywheel.

```python
from graders.langfuse_scorer import LangfuseScorer

scorer = LangfuseScorer()
scorer.submit_grades(task_result)  # Attaches scores to Langfuse trace
```

## Scoring Formula

```
task_score = mean(dimension_scores)
passed = all(dimension_score >= threshold for each dimension)
pass^k = all(trials passed)
```

- Default threshold: **0.75** (configurable per grader)
- Default trials: **3** (pass^3 — all must pass)
- Partial credit: enabled by default (fraction of assertions passed)

## Writing Custom Graders

Subclass `Grader` and implement `grade()`:

```python
from graders.base import Grader, GradeResult

class MyGrader(Grader):
    def grade(self, task_input, task_output, expected=None):
        # Your grading logic here
        score = compute_score(task_output, expected)
        return GradeResult.from_threshold(
            dimension="my_dimension",
            score=score,
            threshold=0.75,
            reasoning="Explanation of score",
        )
```
