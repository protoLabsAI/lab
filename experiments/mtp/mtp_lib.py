"""Shared helpers for the MTP toolkit.

This module is donor-agnostic. It knows the *shape* of a Qwen3.5-family MTP head
(the 15 ``mtp.*`` tensors a native Qwen3.5 checkpoint ships) and nothing about any
particular fine-tune. graft.py / distill.py / validate.sh build on top of it.

Why these exact tensors: a Qwen3.5 MTP head is one ``full_attention`` decoder layer
plus a 2H->H fusion and three RMSNorms, sharing the base model's ``embed_tokens`` and
``lm_head``. See vLLM ``model_executor/models/qwen3_5_mtp.py`` for the reference
serving forward (the thing the distilled head must match).
"""

from __future__ import annotations

import glob
import json
import os
import struct

# The 15 tensors that constitute a Qwen3.5 single-layer MTP head, exactly as a
# native Qwen3.5 checkpoint stores them (top-level ``mtp.*`` prefix). ``num_mtp_layers``
# is 1 for the whole Qwen3.5 family shipped so far (config ``mtp_num_hidden_layers``).
MTP_TENSORS = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]


def snapshot_dir(repo_or_path: str, hf_home: str | None = None) -> str:
    """Resolve a HF repo id or a local path to a concrete snapshot directory.

    Accepts a local dir (returned as-is) or a ``org/name`` repo id (resolved under
    ``$HF_HOME/hub/models--org--name/snapshots/<rev>``).
    """
    if os.path.isdir(repo_or_path) and glob.glob(os.path.join(repo_or_path, "*.safetensors")):
        return repo_or_path
    hf_home = hf_home or os.environ.get("HF_HOME", "/mnt/models/huggingface")
    cache = os.path.join(hf_home, "hub", "models--" + repo_or_path.replace("/", "--"))
    snaps = sorted(glob.glob(os.path.join(cache, "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"no snapshot for {repo_or_path!r} under {cache}")
    # Prefer the snapshot that actually has weights (some revs are config-only).
    for s in reversed(snaps):
        if glob.glob(os.path.join(s, "*.safetensors")):
            return s
    return snaps[-1]


def read_safetensors_header(path: str) -> dict:
    """Return the tensor metadata dict of a .safetensors file (no tensor data read)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def tensor_index(snapshot: str) -> dict[str, tuple[str, tuple[int, ...], str]]:
    """Map tensor name -> (dtype, shape, containing-file) across a snapshot."""
    out: dict[str, tuple[str, tuple[int, ...], str]] = {}
    for sf in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        for k, v in read_safetensors_header(sf).items():
            out[k] = (v["dtype"], tuple(v["shape"]), os.path.basename(sf))
    return out


def validate_graft_compat(donor: str, target: str) -> list[str]:
    """Check that ``donor`` ships the MTP head and ``target`` can host it.

    Returns a list of human-readable problems (empty == compatible). Verifies:
      - donor has all 15 mtp.* tensors,
      - target has none (so we are not clobbering an existing head),
      - target hosts the shared embed_tokens / lm_head the head will tie to,
      - hidden_size agreement (fc out, norms) between donor mtp head and target.
    """
    problems: list[str] = []
    di, ti = tensor_index(donor), tensor_index(target)

    missing = [t for t in MTP_TENSORS if t not in di]
    if missing:
        problems.append(f"donor missing mtp tensors: {missing}")

    present = [t for t in MTP_TENSORS if t in ti]
    if present:
        problems.append(f"target already has mtp tensors (refusing to clobber): {present}")

    shared = ("lm_head.weight", "model.language_model.embed_tokens.weight")
    for s in shared:
        if s not in ti:
            problems.append(f"target missing shared tensor the MTP head ties to: {s}")

    # hidden_size sanity: mtp.fc maps 2H -> H; mtp.norm is (H,).
    if "mtp.fc.weight" in di and "mtp.norm.weight" in di:
        h_out, h_in = di["mtp.fc.weight"][1]
        (h_norm,) = di["mtp.norm.weight"][1]
        if not (h_in == 2 * h_out == 2 * h_norm):
            problems.append(
                f"donor mtp head shape inconsistent: fc={di['mtp.fc.weight'][1]} norm={di['mtp.norm.weight'][1]}"
            )
        # target hidden via lm_head (vocab, H)
        if "lm_head.weight" in ti:
            _, h_target = ti["lm_head.weight"][1]
            if h_target != h_out:
                problems.append(
                    f"hidden_size mismatch: donor mtp H={h_out} vs target H={h_target}"
                )
    return problems
