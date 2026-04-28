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
    ):
        self.dimension = dimension
        self.rubric = rubric or DEFAULT_RUBRIC
        self.model = model
        self.threshold = threshold
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
                max_tokens=500,
            )
            content = response.choices[0].message.content
            if content is None:
                content = ""
            content = content.strip()

            # Parse JSON from response (handle markdown code blocks)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)

            score = float(result.get("score", 0.5))
            reasoning = result.get("reasoning", "")

        except Exception as e:
            score = 0.5
            reasoning = f"Judge error: {e}"

        return GradeResult.from_threshold(
            dimension=self.dimension,
            score=score,
            threshold=self.threshold,
            reasoning=reasoning,
        )
