"""Needle-in-a-Haystack context window test.

Inserts a unique fact at various depths within filler text, then asks the
model to recall it. Tests effective context utilization across lengths.

Usage:
    python -m evals.tasks.context_window.needle_in_haystack \
        --model local --lengths 4096,16384,65536 \
        --gateway-url http://localhost:8000/v1
"""

from __future__ import annotations

import time

import click
from openai import OpenAI

NEEDLE = "The secret code is: PURPLE-TIGER-42"
RETRIEVAL_PROMPT = (
    "What is the secret code mentioned in the text above? "
    "Reply with only the secret code, nothing else."
)

# ~500 tokens of filler per block, repeated as needed.
FILLER_BLOCK = (
    "The history of computing is filled with remarkable innovations. In the early "
    "days, vacuum tubes powered massive machines that filled entire rooms. Engineers "
    "worked tirelessly to improve reliability and reduce the size of these early "
    "computers. The invention of the transistor in 1947 marked a turning point, "
    "enabling smaller and more efficient designs. By the 1960s, integrated circuits "
    "combined multiple transistors on a single chip, paving the way for the "
    "microprocessor revolution. Personal computers emerged in the 1970s, bringing "
    "computational power to homes and offices worldwide. The internet, initially a "
    "military research project, evolved into a global network connecting billions of "
    "people. Cloud computing transformed how businesses deploy and scale applications, "
    "while machine learning algorithms began to tackle problems previously thought to "
    "require human intelligence. Today, large language models represent the latest "
    "chapter in this ongoing story of technological progress, processing vast amounts "
    "of text to generate coherent and contextually relevant responses across an "
    "extraordinary range of tasks and domains.\n\n"
)

# Rough chars-per-token ratio for English prose
CHARS_PER_TOKEN = 4

DEPTHS = [0.10, 0.25, 0.50, 0.75, 0.90]


def build_prompt(target_tokens: int, depth: float) -> str:
    """Build a haystack prompt with the needle inserted at the given depth."""
    # Reserve tokens for needle + retrieval question
    needle_line = f"\n{NEEDLE}\n"
    suffix = f"\n\n{RETRIEVAL_PROMPT}"
    reserved_chars = len(needle_line) + len(suffix)
    filler_chars = max(0, target_tokens * CHARS_PER_TOKEN - reserved_chars)

    # Generate filler
    repeats = (filler_chars // len(FILLER_BLOCK)) + 1
    filler = (FILLER_BLOCK * repeats)[:filler_chars]

    # Insert needle at depth
    insert_pos = int(len(filler) * depth)
    # Snap to paragraph boundary to avoid splitting mid-sentence
    newline_pos = filler.rfind("\n", 0, insert_pos)
    if newline_pos > 0:
        insert_pos = newline_pos

    prompt = filler[:insert_pos] + needle_line + filler[insert_pos:] + suffix
    return prompt


def run_single(
    client: OpenAI, model: str, target_tokens: int, depth: float
) -> tuple[bool, float]:
    """Run one needle test. Returns (passed, latency_seconds)."""
    prompt = build_prompt(target_tokens, depth)

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            # thinking-off: this is pure retrieval, and a thinking budget would consume the
            # whole answer (the orphaned version used max_tokens=50 + thinking-on -> 0/15).
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as e:
        click.echo(f"    ERROR at {target_tokens}tok/{depth:.0%} depth: {e}", err=True)
        return False, time.time() - t0

    latency = time.time() - t0
    passed = "PURPLE-TIGER-42" in answer
    return passed, latency


def print_results(results: dict[tuple[int, float], tuple[bool, float]]) -> None:
    """Print an ASCII table of pass/fail results."""
    lengths = sorted({k[0] for k in results})
    depths = sorted({k[1] for k in results})

    # Header
    depth_headers = [f"{d:.0%}".rjust(7) for d in depths]
    click.echo(f"\n{'Length':>8} | {'  |  '.join(depth_headers)}  |")
    click.echo("-" * (11 + 12 * len(depths)))

    # Rows
    for length in lengths:
        cells = []
        for depth in depths:
            passed, latency = results.get((length, depth), (False, 0.0))
            mark = "PASS" if passed else "FAIL"
            cells.append(f"{mark} {latency:4.1f}s")
        label = f"{length // 1024}K" if length >= 1024 else str(length)
        click.echo(f"{label:>8} | {'  |  '.join(cells)}  |")

    # Summary
    total = len(results)
    passed_count = sum(1 for v in results.values() if v[0])
    click.echo(f"\nTotal: {passed_count}/{total} passed")


@click.command()
@click.option(
    "--model",
    default="local",
    help="Model name to test (use 'local' for the vLLM-served model).",
)
@click.option(
    "--lengths",
    default="4096,8192,16384,32768,65536,131072",
    help="Comma-separated context lengths in tokens.",
)
@click.option(
    "--depths",
    default=None,
    help="Comma-separated depth fractions (default: 0.10,0.25,0.50,0.75,0.90).",
)
@click.option(
    "--gateway-url",
    default="http://localhost:8000/v1",
    help="OpenAI-compatible API base URL.",
)
@click.option("--api-key", envvar=["GATEWAY_API_KEY", "LITELLM_API_KEY"], default="not-needed",
              help="API key (auto-reads GATEWAY_API_KEY/LITELLM_API_KEY; needed for the gateway).")
def main(
    model: str,
    lengths: str,
    depths: str | None,
    gateway_url: str,
    api_key: str,
) -> None:
    """Needle-in-a-Haystack context window test.

    Inserts a secret code at various depths within filler text and checks
    whether the model can retrieve it across different context lengths.
    """
    client = OpenAI(base_url=gateway_url, api_key=api_key)
    length_list = [int(x.strip()) for x in lengths.split(",")]
    depth_list = [float(x.strip()) for x in depths.split(",")] if depths else DEPTHS

    # Resolve model name for local vLLM
    if model == "local":
        models = client.models.list()
        if models.data:
            model = models.data[0].id
            click.echo(f"Resolved 'local' to model: {model}")
        else:
            click.echo("ERROR: No models found at gateway. Is vLLM running?", err=True)
            raise SystemExit(1)

    click.echo(f"Model: {model}")
    click.echo(f"Lengths: {length_list}")
    click.echo(f"Depths: {[f'{d:.0%}' for d in depth_list]}")
    click.echo(f"Gateway: {gateway_url}")
    click.echo()

    results: dict[tuple[int, float], tuple[bool, float]] = {}

    for length in length_list:
        for depth in depth_list:
            label = f"{length // 1024}K" if length >= 1024 else str(length)
            click.echo(f"  Testing {label} @ {depth:.0%} depth ... ", nl=False)
            passed, latency = run_single(client, model, length, depth)
            mark = "PASS" if passed else "FAIL"
            click.echo(f"{mark} ({latency:.1f}s)")
            results[(length, depth)] = (passed, latency)

    print_results(results)


if __name__ == "__main__":
    main()
