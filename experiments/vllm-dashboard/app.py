"""vLLM Management Dashboard — GPU monitoring, model swap, and chat testing."""

from __future__ import annotations

import gradio as gr

from utils import (
    chat_completion,
    extract_config_name,
    get_config_choices,
    get_gpu_info,
    get_vllm_status,
    swap_model,
    tail_log,
)

REFRESH_INTERVAL = 5  # seconds


# ── GPU status rendering ─────────────────────────────────────────────

def render_gpu_cards() -> str:
    """Render GPU status as HTML cards."""
    gpus = get_gpu_info()
    if not gpus:
        return "<p style='color:#888'>Unable to read GPU info</p>"

    status = get_vllm_status()
    cards = []
    for gpu in gpus:
        mem_pct = gpu.memory_pct
        mem_color = "#22c55e" if mem_pct < 50 else "#eab308" if mem_pct < 85 else "#ef4444"
        temp_color = "#22c55e" if gpu.temp_c < 60 else "#eab308" if gpu.temp_c < 80 else "#ef4444"
        power_pct = (gpu.power_draw_w / gpu.power_limit_w * 100) if gpu.power_limit_w > 0 else 0

        card = f"""
        <div style="background:#1a1a2e; border:1px solid #333; border-radius:12px; padding:20px; flex:1; min-width:280px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-size:18px; font-weight:bold; color:#e0e0e0;">GPU {gpu.index}</span>
                <span style="color:{temp_color}; font-size:16px; font-weight:bold;">{gpu.temp_c:.0f}°C</span>
            </div>
            <div style="font-size:12px; color:#888; margin-bottom:16px;">{gpu.name}</div>

            <div style="margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; font-size:13px; color:#aaa; margin-bottom:4px;">
                    <span>VRAM</span>
                    <span>{gpu.memory_used_gb:.1f} / {gpu.memory_total_gb:.1f} GB ({mem_pct:.0f}%)</span>
                </div>
                <div style="background:#333; border-radius:6px; height:12px; overflow:hidden;">
                    <div style="background:{mem_color}; height:100%; width:{mem_pct}%; border-radius:6px; transition:width 0.5s;"></div>
                </div>
            </div>

            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; font-size:13px; color:#aaa; margin-bottom:4px;">
                    <span>Power</span>
                    <span>{gpu.power_draw_w:.0f}W / {gpu.power_limit_w:.0f}W</span>
                </div>
                <div style="background:#333; border-radius:6px; height:8px; overflow:hidden;">
                    <div style="background:#6366f1; height:100%; width:{power_pct}%; border-radius:6px; transition:width 0.5s;"></div>
                </div>
            </div>
        </div>
        """
        cards.append(card)

    # Model status bar
    model_name = status["model"]
    running = status["running"]
    kv = status["kv_cache_pct"]
    reqs = status["requests_running"]
    waiting = status["requests_waiting"]
    status_color = "#22c55e" if running else "#ef4444"
    status_dot = f'<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{status_color}; margin-right:8px;"></span>'

    model_bar = f"""
    <div style="background:#1a1a2e; border:1px solid #333; border-radius:12px; padding:16px; margin-top:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
            <div>
                {status_dot}
                <span style="font-size:16px; font-weight:bold; color:#e0e0e0;">{model_name}</span>
            </div>
            <div style="display:flex; gap:24px; font-size:13px; color:#aaa;">
                <span>KV Cache: <b style="color:#e0e0e0;">{kv:.1f}%</b></span>
                <span>Running: <b style="color:#e0e0e0;">{reqs}</b></span>
                <span>Queued: <b style="color:#e0e0e0;">{waiting}</b></span>
            </div>
        </div>
    </div>
    """

    return f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap;">
        {"".join(cards)}
    </div>
    {model_bar}
    """


# ── Swap handler ─────────────────────────────────────────────────────

def handle_swap(choice: str) -> tuple[str, str]:
    """Handle model swap from dropdown."""
    if not choice:
        return "No model selected.", render_gpu_cards()
    config = extract_config_name(choice)
    result = swap_model(config)
    return result, render_gpu_cards()


# ── Build UI ─────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="#0f0f1a",
    block_background_fill="#1a1a2e",
    block_border_color="#333",
    input_background_fill="#16213e",
    button_primary_background_fill="#6366f1",
)

CSS = ".gradio-container { max-width: 1200px !important; }"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="protoLabs vLLM Dashboard") as app:
        gr.Markdown("# protoLabs vLLM Dashboard")

        # ── GPU Status ────────────────────────────────
        with gr.Row():
            gpu_html = gr.HTML(value=render_gpu_cards, every=REFRESH_INTERVAL)

        # ── Model Control ─────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=get_config_choices(),
                    label="Model Config",
                    interactive=True,
                )
                with gr.Row():
                    swap_btn = gr.Button("Swap Model", variant="primary", size="lg")
                    refresh_btn = gr.Button("Refresh GPU", size="lg")

            with gr.Column(scale=3):
                swap_output = gr.Textbox(
                    label="Swap Output",
                    lines=6,
                    interactive=False,
                                    )

        swap_btn.click(
            fn=handle_swap,
            inputs=[model_dropdown],
            outputs=[swap_output, gpu_html],
        )
        refresh_btn.click(fn=render_gpu_cards, outputs=[gpu_html])

        # ── Chat + Logs tabs ──────────────────────────
        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(
                    label="Test Chat",
                    height=400,
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Message",
                        placeholder="Type a message to test the model...",
                        scale=4,
                        show_label=False,
                    )
                    chat_send = gr.Button("Send", variant="primary", scale=1)
                    chat_clear = gr.Button("Clear", scale=1)

                def handle_chat(message: str, history: list) -> tuple[str, list]:
                    if not message.strip():
                        return "", history
                    # Convert history to format chat_completion expects
                    flat_history = []
                    for msg in history:
                        if isinstance(msg, dict):
                            flat_history.append(msg)

                    response = chat_completion(message, flat_history)
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": response})
                    return "", history

                chat_send.click(
                    fn=handle_chat,
                    inputs=[chat_input, chatbot],
                    outputs=[chat_input, chatbot],
                )
                chat_input.submit(
                    fn=handle_chat,
                    inputs=[chat_input, chatbot],
                    outputs=[chat_input, chatbot],
                )
                chat_clear.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

            with gr.Tab("Logs"):
                log_output = gr.Textbox(
                    value=lambda: tail_log(80),
                    label="vLLM Swap Log (last 80 lines)",
                    lines=20,
                    interactive=False,
                    every=10,
                                    )

    return app


# ── Launch ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=THEME,
        css=CSS,
    )
