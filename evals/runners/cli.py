"""CLI entrypoint for proto-eval."""

import click

from runners.run_claw import main as claw_cmd
from runners.run_custom import main as custom_cmd
from runners.compare import main as compare_cmd
from runners.run_function_call import main as function_call_cmd
from runners.run_rag import main as rag_cmd
from runners.run_profile import main as profile_cmd


@click.group()
def main():
    """protoLabs eval suite — quant + serving lab eval harness."""
    pass


# Suites (consolidated 2026-06-27 — quant+serving focus):
# claw (agentic), custom (coding/reasoning/etc), function-call, rag; profile
# orchestrates; compare diffs runs. wildbench/refusal/inspect/general archived
# to /mnt/data/lab-archive/runners-2026-06-27/.
main.add_command(claw_cmd, "claw")
main.add_command(custom_cmd, "custom")
main.add_command(compare_cmd, "compare")
main.add_command(function_call_cmd, "function-call")
main.add_command(rag_cmd, "rag")
main.add_command(profile_cmd, "profile")


if __name__ == "__main__":
    main()
