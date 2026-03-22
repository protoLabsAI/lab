"""CLI entrypoint for proto-eval."""

import click

from runners.run_claw import main as claw_cmd
from runners.run_custom import main as custom_cmd
from runners.compare import main as compare_cmd
from runners.run_function_call import main as function_call_cmd
from runners.run_rag import main as rag_cmd
from runners.run_profile import main as profile_cmd
from runners.run_wildbench import main as wildbench_cmd


@click.group()
def main():
    """protoLabs eval suite — comprehensive LLM/agent evaluation laboratory."""
    pass


# Core suites
main.add_command(claw_cmd, "claw")
main.add_command(custom_cmd, "custom")
main.add_command(compare_cmd, "compare")
main.add_command(function_call_cmd, "function-call")
main.add_command(rag_cmd, "rag")
main.add_command(profile_cmd, "profile")
main.add_command(wildbench_cmd, "wildbench")

# Optional suites (lazy import — don't break CLI if deps missing)
try:
    from runners.run_inspect import main as inspect_cmd
    main.add_command(inspect_cmd, "inspect")
except ImportError:
    pass

try:
    from runners.run_general import main as general_cmd
    main.add_command(general_cmd, "general")
except ImportError:
    pass


if __name__ == "__main__":
    main()
