"""Pairwise A/B judge — compare two model outputs and pick a winner.

More discriminating than absolute scoring for models that score similarly.
Uses the same gateway API as LLMJudge. Randomizes presentation order to
avoid position bias, and exposes the swap so results can be de-biased.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

PAIRWISE_PROMPT = """You are an expert evaluator performing a blind comparison.

## Dimension
{dimension}

{rubric_section}

## Prompt Given to Both Models
{prompt}

## Response A
{response_a}

## Response B
{response_b}

## Instructions
Compare Response A and Response B on the dimension above.
- Pick the better response, or declare a tie if they are truly equal.
- Provide a confidence score from 0.5 (coin-flip) to 1.0 (absolutely certain).
- Explain your reasoning briefly.

Respond with JSON only:
{{"winner": "A" | "B" | "tie", "confidence": <float 0.5-1.0>, "reasoning": "<brief explanation>"}}
"""


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison between two outputs."""

    winner: str  # "A", "B", or "tie" (in caller's original order)
    confidence: float  # 0.5 - 1.0
    reasoning: str = ""
    swapped: bool = False  # True if presentation order was swapped
    dimension: str = "overall_quality"
    metadata: dict[str, Any] = field(default_factory=dict)


class PairwiseJudge:
    """Compare two model outputs for the same prompt and pick a winner.

    More discriminating than absolute scoring for models that score similarly.
    Uses the same gateway API as LLMJudge.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:4000/v1",
        api_key: str | None = None,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def judge(
        self,
        prompt: str,
        output_a: str,
        output_b: str,
        dimension: str = "overall_quality",
        rubric: str | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> PairwiseResult:
        """Returns winner ('A', 'B', or 'tie'), confidence (0-1), and reasoning.

        Internally randomizes presentation order to avoid position bias.
        The returned winner always refers to the caller's original A/B.
        """
        # Randomize order to eliminate position bias
        swapped = random.random() < 0.5
        if swapped:
            shown_a, shown_b = output_b, output_a
        else:
            shown_a, shown_b = output_a, output_b

        rubric_section = f"## Rubric\n{rubric}" if rubric else ""

        judge_prompt = PAIRWISE_PROMPT.format(
            dimension=dimension,
            rubric_section=rubric_section,
            prompt=prompt,
            response_a=shown_a,
            response_b=shown_b,
        )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()

            # Parse JSON (handle markdown code blocks)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)

            raw_winner = str(result.get("winner", "tie")).upper()
            confidence = max(0.5, min(1.0, float(result.get("confidence", 0.5))))
            reasoning = result.get("reasoning", "")

        except Exception as e:
            return PairwiseResult(
                winner="tie",
                confidence=0.5,
                reasoning=f"Judge error: {e}",
                swapped=swapped,
                dimension=dimension,
            )

        # Map the judge's winner back to the caller's original order
        if raw_winner == "TIE":
            winner = "tie"
        elif swapped:
            winner = "B" if raw_winner == "A" else "A"
        else:
            winner = raw_winner if raw_winner in ("A", "B") else "tie"

        return PairwiseResult(
            winner=winner,
            confidence=confidence,
            reasoning=reasoning,
            swapped=swapped,
            dimension=dimension,
            metadata={"model": model, "raw_winner": raw_winner},
        )
