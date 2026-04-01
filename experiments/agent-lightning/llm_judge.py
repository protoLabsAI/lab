"""LLM Judge for protoResearcher response evaluation.

Replaces deterministic pattern matching with multi-dimensional LLM scoring.
Uses Claude Sonnet 4.6 via ava gateway as the judge model.

Designed for use in both APO optimization and A/B eval comparisons.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://100.101.189.45:4000/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")

JUDGE_PROMPT = """\
You are an expert evaluator scoring an AI research assistant's response.

## Task Given to the Assistant
{task_prompt}

## Expected Behavior
- Category: {category}
- Expected tools: {expected_tools}
- Key topics that should appear: {expected_patterns}

## Assistant's Response
{response}

## Scoring Instructions
Rate the response on each dimension (0.0 to 1.0). Be strict but fair.

1. **relevance** — Does the response directly address the task? (0 = off-topic, 1 = precisely on-target)
2. **completeness** — Does it cover the full scope of what was asked? (0 = missing most content, 1 = thorough)
3. **structure** — Is it well-organized with clear formatting? (0 = wall of text, 1 = excellent structure)
4. **accuracy** — Are claims factually correct and well-grounded? (0 = hallucinated, 1 = accurate)
5. **actionability** — Does it provide practical value a researcher can act on? (0 = vague, 1 = actionable)

Special cases:
- If the task is an edge case (empty input), score based on graceful handling.
- If the task requires tool use and the response shows no evidence of tool output, penalize completeness.
- If the response is an error message, score 0 on all dimensions.

Output ONLY valid JSON, no markdown fences, no extra text:
{{"relevance": 0.0, "completeness": 0.0, "structure": 0.0, "accuracy": 0.0, "actionability": 0.0, "reasoning": "brief explanation"}}"""


@dataclass
class JudgeScore:
    relevance: float
    completeness: float
    structure: float
    accuracy: float
    actionability: float
    reasoning: str
    composite: float  # weighted average

    def to_dict(self) -> dict:
        return {
            "relevance": self.relevance,
            "completeness": self.completeness,
            "structure": self.structure,
            "accuracy": self.accuracy,
            "actionability": self.actionability,
            "reasoning": self.reasoning,
            "composite": self.composite,
        }


# Weights for composite score
WEIGHTS = {
    "relevance": 0.25,
    "completeness": 0.25,
    "structure": 0.15,
    "accuracy": 0.20,
    "actionability": 0.15,
}


def _compute_composite(scores: dict[str, float]) -> float:
    total = sum(WEIGHTS[k] * scores.get(k, 0.0) for k in WEIGHTS)
    return round(total, 3)


def _parse_judge_output(raw: str) -> dict:
    """Extract JSON from judge response, tolerating markdown fences and preamble."""
    # Try to find a JSON object anywhere in the text
    match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in judge output: {raw[:200]}")


def _make_client() -> AsyncOpenAI:
    api_key = os.environ.get("GATEWAY_API_KEY", "not-needed")
    return AsyncOpenAI(base_url=GATEWAY_URL, api_key=api_key)


async def judge_response(
    task: dict,
    response: str,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
) -> JudgeScore:
    """Score a single response using the LLM judge.

    Args:
        task: Task dict with id, prompt, category, expected_patterns, expected_tools.
        response: The agent's response text.
        client: OpenAI client (created if None).
        model: Model ID to use (defaults to JUDGE_MODEL).

    Returns:
        JudgeScore with per-dimension scores and composite.
    """
    if client is None:
        client = _make_client()
    if model is None:
        model = JUDGE_MODEL

    prompt = JUDGE_PROMPT.format(
        task_prompt=task.get("prompt", "(empty)") or "(empty)",
        category=task.get("category", "unknown"),
        expected_tools=", ".join(task.get("expected_tools", [])) or "none",
        expected_patterns=", ".join(task.get("expected_patterns", [])) or "none",
        response=response[:3000] if response else "(empty response)",
    )

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or ""
        scores = _parse_judge_output(raw)

        dims = {}
        for k in WEIGHTS:
            v = scores.get(k, 0.0)
            dims[k] = max(0.0, min(1.0, float(v)))

        return JudgeScore(
            **dims,
            reasoning=scores.get("reasoning", ""),
            composite=_compute_composite(dims),
        )
    except Exception as e:
        return JudgeScore(
            relevance=0.0,
            completeness=0.0,
            structure=0.0,
            accuracy=0.0,
            actionability=0.0,
            reasoning=f"Judge error: {e}",
            composite=0.0,
        )


async def judge_batch(
    tasks_and_responses: list[tuple[dict, str]],
    client: AsyncOpenAI | None = None,
    model: str | None = None,
) -> list[JudgeScore]:
    """Score multiple (task, response) pairs sequentially."""
    if client is None:
        client = _make_client()

    scores = []
    for task, response in tasks_and_responses:
        score = await judge_response(task, response, client=client, model=model)
        scores.append(score)
    return scores
