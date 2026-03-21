"""RAG evaluation graders — retrieval-augmented generation quality.

Four dimensions, all using LLM-as-judge via the gateway:
- Groundedness: Is the answer supported by the retrieved context?
- Context Relevance: Are the retrieved chunks relevant to the question?
- Answer Relevance: Does the answer address the question?
- Faithfulness: Does the answer avoid hallucinating beyond context?
"""

from __future__ import annotations

from graders.llm_judge import LLMJudge

GROUNDEDNESS_RUBRIC = """You are evaluating whether an AI assistant's answer is grounded in the provided context.

## Question
{input}

## Retrieved Context
{expected}

## Assistant's Answer
{output}

## Scoring
- 1.0: Every claim in the answer is directly supported by the context
- 0.75: Most claims are supported, minor unsupported details
- 0.5: Some claims are supported but significant information is added
- 0.25: Few claims are supported, mostly fabricated content
- 0.0: Answer contradicts the context or is entirely unsupported

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

CONTEXT_RELEVANCE_RUBRIC = """You are evaluating whether retrieved context chunks are relevant to the question.

## Question
{input}

## Retrieved Context
{expected}

## Scoring
- 1.0: All context chunks are highly relevant to answering the question
- 0.75: Most chunks are relevant, some tangential
- 0.5: Mixed — some relevant, some irrelevant chunks
- 0.25: Mostly irrelevant context
- 0.0: Context is completely unrelated to the question

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

ANSWER_RELEVANCE_RUBRIC = """You are evaluating whether the answer addresses the question asked.

## Question
{input}

## Assistant's Answer
{output}

## Scoring
- 1.0: Answer directly and completely addresses the question
- 0.75: Answer mostly addresses the question, minor gaps
- 0.5: Answer partially addresses the question
- 0.25: Answer is tangentially related
- 0.0: Answer does not address the question at all

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

FAITHFULNESS_RUBRIC = """You are evaluating whether the answer stays faithful to the provided context without hallucinating.

## Question
{input}

## Retrieved Context
{expected}

## Assistant's Answer
{output}

## Scoring
- 1.0: Answer only contains information from the context, no hallucination
- 0.75: Minor extrapolation but no factual hallucination
- 0.5: Some hallucinated facts mixed with context-based information
- 0.25: Significant hallucination, unreliable answer
- 0.0: Entirely hallucinated, no basis in context

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""


def groundedness_judge(**kwargs) -> LLMJudge:
    return LLMJudge(dimension="groundedness", rubric=GROUNDEDNESS_RUBRIC, **kwargs)


def context_relevance_judge(**kwargs) -> LLMJudge:
    return LLMJudge(dimension="context_relevance", rubric=CONTEXT_RELEVANCE_RUBRIC, **kwargs)


def answer_relevance_judge(**kwargs) -> LLMJudge:
    return LLMJudge(dimension="answer_relevance", rubric=ANSWER_RELEVANCE_RUBRIC, **kwargs)


def faithfulness_judge(**kwargs) -> LLMJudge:
    return LLMJudge(dimension="faithfulness", rubric=FAITHFULNESS_RUBRIC, **kwargs)


def all_rag_judges(**kwargs) -> list[LLMJudge]:
    """Return all four RAG judges for comprehensive evaluation."""
    return [
        groundedness_judge(**kwargs),
        context_relevance_judge(**kwargs),
        answer_relevance_judge(**kwargs),
        faithfulness_judge(**kwargs),
    ]
