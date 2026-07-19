"""protoLabs.nodes — ComfyUI custom nodes wiring workflows into the protoLabs ecosystem
(LiteLLM gateway lanes, protoAgent-style OpenAI endpoints, Fish TTS, Whisper).

Install: symlink this directory into ComfyUI/custom_nodes/, e.g.
  ln -s ~/dev/lab/infra/protolab-nodes ~/dev/ComfyUI/custom_nodes/protolab-nodes

Lives in the lab monorepo (not in ComfyUI/custom_nodes directly) so it stays under
version control — same pattern as protolab-ace.
"""
from .nodes_llm import NODE_CLASS_MAPPINGS as _N1
from .nodes_llm import NODE_DISPLAY_NAME_MAPPINGS as _D1
from .nodes_prompt import NODE_CLASS_MAPPINGS as _N2
from .nodes_prompt import NODE_DISPLAY_NAME_MAPPINGS as _D2
from .nodes_audio import NODE_CLASS_MAPPINGS as _N3
from .nodes_audio import NODE_DISPLAY_NAME_MAPPINGS as _D3

NODE_CLASS_MAPPINGS = {**_N1, **_N2, **_N3}
NODE_DISPLAY_NAME_MAPPINGS = {**_D1, **_D2, **_D3}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
