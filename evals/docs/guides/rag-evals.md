# RAG Evaluation

Evaluates retrieval-augmented generation quality across four dimensions using
LLM-as-judge scoring via the gateway.

## Dimensions

| Dimension | Question Answered |
|-----------|-------------------|
| **Groundedness** | Is every claim in the answer supported by the context? |
| **Context Relevance** | Are the retrieved chunks relevant to the question? |
| **Answer Relevance** | Does the answer actually address the question? |
| **Faithfulness** | Does the answer avoid hallucinating beyond the context? |

## Running

```bash
# Run with default test cases
./run.sh rag --model local

# Specify judge model (cloud model recommended for accuracy)
./run.sh rag --model local --judge-model claude-sonnet-4-6

# Custom test file
./run.sh rag --model local --test-file path/to/tests.yaml
```

## Test Case Format

```yaml
tests:
  - question: "What is the max VRAM on the RTX PRO 6000?"
    context: |
      The NVIDIA RTX PRO 6000 Blackwell GPU features 96 GB of GDDR7 VRAM.
    expected_answer: "96 GB of GDDR7 VRAM"
```

The `context` field simulates retrieved chunks. The `expected_answer` is optional
but helps the judge calibrate.

## Key Test Scenarios

1. **Supported answer** — context contains the answer, model should use it
2. **Irrelevant context** — context is unrelated, model should say "not in context"
3. **Partial context** — some info available, model should answer what it can
4. **Contradictory context** — conflicting chunks, model should note the conflict
5. **Multi-hop** — answer requires combining info from multiple context chunks
