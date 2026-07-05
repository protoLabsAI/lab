"""LLM-as-Judge grader — model-based rubric scoring via gateway.

Best practices from Anthropic:
- Grade each dimension in isolation (separate judge calls)
- Provide a clear rubric with score anchors
- Give the judge a way out ("Unknown" / 0.5 when unsure)
- Calibrate against human judgment periodically
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from openai import OpenAI

from graders.base import Grader, GradeResult

DEFAULT_RUBRIC = """You are an expert evaluator. Score the agent's output on the dimension: {dimension}.

## Rubric
- 1.0: Excellent — fully meets criteria, no issues
- 0.75: Good — meets most criteria, minor issues
- 0.5: Partial — meets some criteria, notable gaps
- 0.25: Poor — barely addresses criteria
- 0.0: Fail — does not address criteria at all

If you cannot determine a score, respond with 0.5.

## Task Input
{input}

## Expected Output (if available)
{expected}

## Agent Output
{output}

## Instructions
Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""


class LLMJudge(Grader):
    """Score a dimension using an LLM judge via the gateway."""

    def __init__(
        self,
        dimension: str,
        rubric: str | None = None,
        model: str = "claude-sonnet-4-6",
        base_url: str = os.environ.get("JUDGE_GATEWAY_URL", "http://ava:4000/v1"),
        api_key: str | None = None,
        threshold: float = 0.75,
        raise_on_error: bool = False,
    ):
        self.dimension = dimension
        self.rubric = rubric or DEFAULT_RUBRIC
        self.model = model
        self.threshold = threshold
        # RL callers pass raise_on_error=True: a judge failure must NOT become a usable
        # 0.5 reward (a policy could farm it by inducing judge errors). Raising lets the
        # RL loop drop the sample (zero-and-exclude) instead. Eval callers keep the
        # default — a loud stderr 0.5 is a deliberate measurement choice there.
        self.raise_on_error = raise_on_error
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def grade(self, task_input: dict, task_output: dict, expected: dict | None = None) -> GradeResult:
        input_str = json.dumps(task_input, indent=2, default=str)
        output_str = json.dumps(task_output, indent=2, default=str)
        expected_str = json.dumps(expected, indent=2, default=str) if expected else "N/A"

        # If the custom rubric has template placeholders, use them
        if "{input}" in self.rubric or "{output}" in self.rubric:
            prompt = self.rubric.format(
                dimension=self.dimension,
                input=input_str,
                expected=expected_str,
                output=output_str,
            )
        else:
            # Custom rubric without placeholders — wrap with context
            prompt = (
                f"You are evaluating an AI's output on the dimension: {self.dimension}.\n\n"
                f"## Rubric\n{self.rubric}\n\n"
                f"## Task Input\n{input_str}\n\n"
                f"## AI Output\n{output_str}\n\n"
                f"## Expected (if available)\n{expected_str}\n\n"
                "Respond with JSON only:\n"
                '{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}'
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,  # reasoning-judge headroom (think block + answer); see FOCUS.md
                # Judge must return clean JSON — disable thinking on reasoning-model judges
                # (a thinking model's wrapped output breaks json.loads -> silent 0.5 fallback).
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = (response.choices[0].message.content or "").strip()

            # Robust extraction: drop any leaked think block, strip code fences,
            # then pull the first {...} object out of surrounding prose.
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                content = m.group(0)
            result = json.loads(content)

            score = float(result.get("score", 0.5))
            reasoning = result.get("reasoning", "")

        except Exception as e:
            # Do NOT silently bury judge failures as 0.5 — surface them so a broken
            # judge can't masquerade as a wall of 0.5 scores (the recurring footgun).
            if self.raise_on_error:
                # RL path: propagate so the caller excludes the sample rather than
                # rewarding a judge error with a farmable 0.5.
                raise
            score = 0.5
            reasoning = f"Judge error (FALLBACK 0.5): {e}"
            print(f"[llm_judge] {self.dimension}: judge call/parse failed -> 0.5 fallback: {e}",
                  file=sys.stderr)

        return GradeResult.from_threshold(
            dimension=self.dimension,
            score=score,
            threshold=self.threshold,
            reasoning=reasoning,
        )
