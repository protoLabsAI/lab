"""DeepEval metric wrappers for our grading framework.

Provides G-Eval summarization scoring, hallucination detection,
and faithfulness checking using the deepeval library with our
vLLM/gateway as the judge model.

Usage:
    from graders.deepeval_metrics import GEvalSummarization, HallucinationGrader

    grader = GEvalSummarization(model="claude-sonnet-4-6")
    result = grader.grade(source_text, summary)

    grader = HallucinationGrader(model="claude-sonnet-4-6")
    result = grader.grade(context, output)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from graders.base import GradeResult


def _get_deepeval_model(model: str = "claude-sonnet-4-6"):
    """Create a DeepEval model pointed at our gateway."""
    from deepeval.models import DeepEvalBaseLLM
    from openai import OpenAI

    gateway_url = os.environ.get("JUDGE_GATEWAY_URL", "http://ava:4000/v1")
    api_key = os.environ.get("GATEWAY_API_KEY", "not-needed")

    class GatewayModel(DeepEvalBaseLLM):
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.client = OpenAI(base_url=gateway_url, api_key=api_key)

        def load_model(self):
            return self.model_name

        def generate(self, prompt: str, **kwargs) -> str:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0,
            )
            return resp.choices[0].message.content or ""

        async def a_generate(self, prompt: str, **kwargs) -> str:
            return self.generate(prompt, **kwargs)

        def get_model_name(self) -> str:
            return self.model_name

    return GatewayModel(model)


class GEvalSummarization:
    """G-Eval style summarization scoring (Microsoft, 2023).

    Evaluates on 4 dimensions: coherence, consistency, fluency, relevance.
    Uses chain-of-thought prompting for higher correlation with human judgment.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def grade(
        self, source: str, summary: str, dimensions: list[str] | None = None
    ) -> list[GradeResult]:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        dims = dimensions or ["coherence", "consistency", "fluency", "relevance"]
        judge = _get_deepeval_model(self.model)
        results = []

        for dim in dims:
            criteria = {
                "coherence": "the summary should be well-structured and well-organized, with ideas flowing logically",
                "consistency": "the summary should contain only statements entailed by the source document, with no hallucinated facts",
                "fluency": "the summary should be grammatically correct, easy to read, and free of formatting issues",
                "relevance": "the summary should capture the key points of the source document without including unnecessary details",
            }.get(dim, dim)

            metric = GEval(
                name=dim,
                criteria=criteria,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                model=judge,
                threshold=0.75,
            )

            test_case = LLMTestCase(input=source, actual_output=summary)
            metric.measure(test_case)

            results.append(
                GradeResult(
                    dimension=dim,
                    score=metric.score,
                    passed=metric.score >= 0.75,
                    reasoning=metric.reason or "",
                )
            )

        return results


class HallucinationGrader:
    """Detect hallucinated claims in model output given a context.

    Decomposes the output into atomic claims and verifies each
    against the provided context.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def grade(self, context: str | list[str], output: str) -> GradeResult:
        from deepeval.metrics import HallucinationMetric
        from deepeval.test_case import LLMTestCase

        contexts = [context] if isinstance(context, str) else context
        judge = _get_deepeval_model(self.model)

        metric = HallucinationMetric(model=judge, threshold=0.75)
        test_case = LLMTestCase(
            input="",
            actual_output=output,
            context=contexts,
        )
        metric.measure(test_case)

        return GradeResult(
            dimension="hallucination",
            score=1.0 - metric.score,  # deepeval returns hallucination rate, we want faithfulness
            passed=(1.0 - metric.score) >= 0.75,
            reasoning=metric.reason or "",
        )


class FaithfulnessGrader:
    """RAGAS-style faithfulness scoring with atomic claim decomposition.

    Breaks the answer into individual claims, then checks each claim
    against the provided context for support.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def grade(self, question: str, context: str | list[str], answer: str) -> GradeResult:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        contexts = [context] if isinstance(context, str) else context
        judge = _get_deepeval_model(self.model)

        metric = FaithfulnessMetric(model=judge, threshold=0.75)
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
        )
        metric.measure(test_case)

        return GradeResult(
            dimension="faithfulness",
            score=metric.score,
            passed=metric.score >= 0.75,
            reasoning=metric.reason or "",
        )
