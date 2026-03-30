"""Harmony message format for GPT-OSS / Context-1 models.

Implements the OpenAI Harmony chat format used by chromadb/context-1.
Reference: https://github.com/openai/harmony
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Special tokens
START = "<|start|>"
END = "<|end|>"
MESSAGE = "<|message|>"
CHANNEL = "<|channel|>"
CALL = "<|call|>"
RETURN = "<|return|>"
CONSTRAIN = "<|constrain|>"
ENDOFPROMPT = "<|endofprompt|>"

# Stop tokens for generation
STOP_TOKENS = [CALL, RETURN]


@dataclass
class ToolDef:
    """Tool definition in TypeScript-style format."""

    name: str
    description: str
    params: dict[str, dict] = field(default_factory=dict)  # name -> {type, desc, optional}

    def to_typescript(self) -> str:
        lines = []
        lines.append(f"// {self.description}")
        if not self.params:
            lines.append(f"type {self.name} = () => any;")
        else:
            lines.append(f"type {self.name} = (_: {{")
            param_lines = []
            for pname, pinfo in self.params.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("desc", "")
                optional = "?" if pinfo.get("optional") else ""
                comment = f" // {pdesc}" if pdesc else ""
                param_lines.append(f"{pname}{optional}: {ptype},{comment}")
            lines.extend(param_lines)
            lines.append("}) => any;")
        return "\n".join(lines)


def format_tools_block(tools: list[ToolDef]) -> str:
    """Format tool definitions as a TypeScript namespace block."""
    parts = ["# Tools", "", "## functions", "", "namespace functions {", ""]
    for tool in tools:
        parts.append(tool.to_typescript())
        parts.append("")
    parts.append("} // namespace functions")
    return "\n".join(parts)


def format_system_message(content: str, has_tools: bool = False) -> str:
    """Format a system message."""
    if has_tools:
        content += (
            "\n\n# Valid channels: analysis, commentary, final. "
            "Channel must be included for every message.\n"
            "Calls to these tools must go to the commentary channel: 'functions'."
        )
    return f"{START}system{MESSAGE}{content}{END}"


def format_developer_message(content: str) -> str:
    """Format a developer message (tool definitions, instructions)."""
    return f"{START}developer{MESSAGE}{content}{END}"


def format_user_message(content: str) -> str:
    """Format a user message."""
    return f"{START}user{MESSAGE}{content}{END}"


def format_assistant_prefix() -> str:
    """Start of assistant generation (no closing token — model generates from here)."""
    return f"{START}assistant"


def format_tool_result(tool_name: str, result: str) -> str:
    """Format a tool result to feed back to the model."""
    return f"{START}functions.{tool_name} to=assistant{CHANNEL}commentary{MESSAGE}{result}{END}"


def format_token_usage(used: int, budget: int) -> str:
    """Append token usage indicator."""
    return f"\n[Token usage: {used:,}/{budget:,}]"


def parse_assistant_output(text: str) -> list[dict]:
    """Parse model output into structured messages.

    Returns list of dicts with keys: channel, content, tool_call (optional).
    A tool_call dict has: name, arguments (str).
    """
    messages = []
    # Split on message boundaries — the model can generate multiple
    # <|start|>assistant segments in one pass
    parts = text.split(f"{START}assistant")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract channel and tool target.
        # Harmony format: the `to=functions.X` can appear in either:
        #   1. Role section (before <|channel|>): " to=functions.X<|channel|>commentary..."
        #   2. Channel section (after <|channel|>): "<|channel|>commentary to=functions.X..."
        channel = None
        tool_target = None
        content = part

        # Check for to= in role section (before <|channel|>)
        role_prefix = part.split(CHANNEL, 1)[0] if CHANNEL in part else part.split(MESSAGE, 1)[0] if MESSAGE in part else ""
        if "to=" in role_prefix:
            _, _, tool_target = role_prefix.partition("to=")
            tool_target = tool_target.strip()

        if CHANNEL in part:
            header, _, content = part.partition(MESSAGE)
            channel_part = header.split(CHANNEL, 1)[1].strip()
            # Also check for to= in channel section
            if " to=" in channel_part and not tool_target:
                channel_name, _, target = channel_part.partition(" to=")
                channel = channel_name.strip()
                tool_target = target.strip()
            else:
                # Strip any constrain markers from channel name
                channel = channel_part.split(CONSTRAIN)[0].strip()
        elif MESSAGE in part:
            _, _, content = part.partition(MESSAGE)

        # Clean trailing tokens
        for tok in [END, CALL, RETURN]:
            content = content.replace(tok, "")
        content = content.strip()

        # Strip constrain marker from tool calls
        if CONSTRAIN in content:
            content = content.split(f"{CONSTRAIN}json", 1)[-1].strip()

        msg: dict = {"channel": channel, "content": content}

        if tool_target and tool_target.startswith("functions."):
            fn_name = tool_target[len("functions."):]
            # Handle constrain marker in the channel header too
            if " " in fn_name:
                fn_name = fn_name.split()[0]
            msg["tool_call"] = {"name": fn_name, "arguments": content}

        messages.append(msg)

    return messages


def build_prompt(
    system: str,
    tools: list[ToolDef],
    history: list[dict],
    token_usage: tuple[int, int] | None = None,
) -> str:
    """Build a complete prompt from system message, tools, and conversation history.

    history entries are dicts with:
      - role: "user" | "assistant" | "tool_result"
      - content: str
      - channel: str (for assistant messages)
      - tool_name: str (for tool_result)
      - tool_call: dict (for assistant tool calls)
    """
    parts = [format_system_message(system, has_tools=bool(tools))]

    if tools:
        parts.append(format_developer_message(format_tools_block(tools)))

    for msg in history:
        role = msg["role"]
        if role == "user":
            parts.append(format_user_message(msg["content"]))
        elif role == "assistant":
            channel = msg.get("channel", "final")
            content = msg["content"]
            if msg.get("tool_call"):
                tc = msg["tool_call"]
                parts.append(
                    f"{START}assistant{CHANNEL}{channel} to=functions.{tc['name']} "
                    f"{CONSTRAIN}json{MESSAGE}{tc['arguments']}{CALL}"
                )
            else:
                end_tok = END
                parts.append(
                    f"{START}assistant{CHANNEL}{channel}{MESSAGE}{content}{end_tok}"
                )
        elif role == "tool_result":
            parts.append(format_tool_result(msg["tool_name"], msg["content"]))

    # Add token usage to the last tool result if provided
    if token_usage:
        used, budget = token_usage
        parts.append(format_token_usage(used, budget))

    # Open assistant generation
    parts.append(format_assistant_prefix())

    return "".join(parts)
