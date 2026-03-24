"""
Qwen3-Omni Demo — adapted for local vLLM-Omni endpoint.

Original source: https://huggingface.co/spaces/Qwen/Qwen3-Omni-Demo
Modified to use a local vLLM-compatible OpenAI API at http://localhost:8091
instead of Alibaba Cloud DashScope + OSS.

Changes from upstream:
  - Removed OSSReader and all oss2/Alibaba Cloud dependencies
  - Media files are base64-encoded inline instead of uploaded to OSS
  - OpenAI client points at LOCAL_API_BASE (default: http://localhost:8091/v1)
  - Model name defaults to LOCAL_MODEL_NAME env var or "qwen3-omni"
  - API key defaults to "EMPTY" (vLLM doesn't require one)
  - Removed torch import (not needed for API-only client)
"""

import io
import os
import base64
import subprocess
import uuid

from argparse import ArgumentParser

import gradio as gr
import gradio.processing_utils as processing_utils
import numpy as np
import soundfile as sf
from gradio_client import utils as client_utils
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
LOCAL_API_BASE = os.environ.get("LOCAL_API_BASE", "http://localhost:8091/v1")
LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "qwen3-omni")
API_KEY = os.environ.get("API_KEY", "EMPTY")

WAV_SAMPLE_RATE = int(os.environ.get("WAV_SAMPLE_RATE", 16000))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_base64(file_path: str) -> str:
    """Read a file and return its base64 encoding."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def mime_to_data_uri(file_path: str) -> str:
    """Return a data-URI string for a local file."""
    mime = client_utils.get_mimetype(file_path)
    b64 = encode_base64(file_path)
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Model loader (just an OpenAI client pointed at the local endpoint)
# ---------------------------------------------------------------------------
def _load_model_processor(args):
    model = OpenAI(
        api_key=API_KEY,
        base_url=LOCAL_API_BASE,
    )
    return model, None


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _launch_demo(args, model, processor):
    VOICE_OPTIONS = {
        "Cherry / 芊悦": "Cherry",
        "Serena / 苏瑶": "Serena",
        "Ethan / 晨煦": "Ethan",
        "Chelsie / 千雪": "Chelsie",
        "Momo / 茉兔": "Momo",
        "Vivian / 十三": "Vivian",
        "Moon / 月白": "Moon",
        "Maia / 四月": "Maia",
        "Kai / 凯": "Kai",
        "Nofish / 不吃鱼": "Nofish",
        "Bella / 萌宝": "Bella",
        "Jennifer / 詹妮弗": "Jennifer",
        "Ryan / 甜茶": "Ryan",
        "Katerina / 卡捷琳娜": "Katerina",
        "Aiden / 艾登": "Aiden",
        "Bodega / 西班牙语-博德加": "Bodega",
        "Alek / 俄语-阿列克": "Alek",
        "Dolce / 意大利语-多尔切": "Dolce",
        "Sohee / 韩语-素熙": "Sohee",
        "Ono Anna / 日语-小野杏": "Ono Anna",
        "Lenn / 德语-莱恩": "Lenn",
        "Sonrisa / 西班牙语拉美-索尼莎": "Sonrisa",
        "Emilien / 法语-埃米尔安": "Emilien",
        "Andre / 葡萄牙语欧-安德雷": "Andre",
        "Radio Gol / 葡萄牙语巴-拉迪奥·戈尔": "Radio Gol",
        "Eldric Sage / 精品百人-沧明子": "Eldric Sage",
        "Mia / 精品百人-乖小妹": "Mia",
        "Mochi / 精品百人-沙小弥": "Mochi",
        "Bellona / 精品百人-燕铮莺": "Bellona",
        "Vincent / 精品百人-田叔": "Vincent",
        "Bunny / 精品百人-萌小姬": "Bunny",
        "Neil / 精品百人-阿闻": "Neil",
        "Elias / 墨讲师": "Elias",
        "Arthur / 精品百人-徐大爷": "Arthur",
        "Nini / 精品百人-邻家妹妹": "Nini",
        "Ebona / 精品百人-诡婆婆": "Ebona",
        "Seren / 精品百人-小婉": "Seren",
        "Pip / 精品百人-调皮小新": "Pip",
        "Stella / 精品百人-美少女阿月": "Stella",
        "Li / 南京-老李": "Li",
        "Marcus / 陕西-秦川": "Marcus",
        "Roy / 闽南-阿杰": "Roy",
        "Peter / 天津-李彼得": "Peter",
        "Eric / 四川-程川": "Eric",
        "Rocky / 粤语-阿强": "Rocky",
        "Kiki / 粤语-阿清": "Kiki",
        "Sunny / 四川-晴儿": "Sunny",
        "Jada / 上海-阿珍": "Jada",
        "Dylan / 北京-晓东": "Dylan",
    }
    DEFAULT_VOICE = "Cherry / 芊悦"

    default_system_prompt = ''

    language = args.ui_language

    def get_text(text: str, cn_text: str):
        if language == 'en':
            return text
        if language == 'zh':
            return cn_text
        return text

    def to_mp4(path):
        if path and path.endswith(".webm"):
            mp4_path = path.replace(".webm", ".mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", path,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-tune", "fastdecode",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-threads", "0",
                    "-f", "mp4",
                    mp4_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            return mp4_path
        return path

    def format_history(history: list, system_prompt: str):
        print(history)
        messages = []
        if system_prompt != "":
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        current_user_content = []

        for item in history:
            role = item['role']
            content = item['content']

            if role != "user":
                if current_user_content:
                    messages.append({
                        "role": "user",
                        "content": current_user_content
                    })
                    current_user_content = []

                if isinstance(content, str):
                    messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    })
                else:
                    pass
                continue

            if isinstance(content, str):
                current_user_content.append({"type": "text", "text": content})
            elif isinstance(content, (list, tuple)):
                for file_path in content:
                    mime_type = client_utils.get_mimetype(file_path)
                    media_type = None

                    if mime_type.startswith("image"):
                        media_type = "image_url"
                    elif mime_type.startswith("video"):
                        media_type = "video_url"
                        file_path = to_mp4(file_path)
                    elif mime_type.startswith("audio"):
                        media_type = "input_audio"

                    if media_type:
                        # Encode media as base64 data URI for local endpoint
                        data_uri = mime_to_data_uri(file_path)

                        if media_type == "input_audio":
                            current_user_content.append({
                                "type": "input_audio",
                                "input_audio": {
                                    "data": data_uri,
                                    "format": "wav",
                                },
                            })
                        elif media_type == "image_url":
                            current_user_content.append({
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            })
                        elif media_type == "video_url":
                            current_user_content.append({
                                "type": "video_url",
                                "video_url": {"url": data_uri},
                            })
                    else:
                        current_user_content.append({
                            "type": "text",
                            "text": file_path
                        })

        if current_user_content:
            media_items = []
            text_items = []

            for item in current_user_content:
                if item["type"] == "text":
                    text_items.append(item)
                else:
                    media_items.append(item)

            messages.append({
                "role": "user",
                "content": media_items + text_items
            })

        return messages

    def predict(messages,
                voice_choice=DEFAULT_VOICE,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                return_audio=False,
                enable_thinking=False):
        if enable_thinking:
            return_audio = False
        if return_audio:
            completion = model.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=messages,
                modalities=["text", "audio"],
                audio={
                    "voice": VOICE_OPTIONS[voice_choice],
                    "format": "wav"
                },
                extra_body={
                    'enable_thinking': False,
                    "top_k": top_k
                },
                stream_options={"include_usage": True},
                stream=True,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            completion = model.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=messages,
                modalities=["text"],
                extra_body={
                    'enable_thinking': enable_thinking,
                    "top_k": top_k
                },
                stream_options={"include_usage": True},
                stream=True,
                temperature=temperature,
                top_p=top_p,
            )
        audio_string = ""
        output_text = ""
        reasoning_content = "<think>\n\n"
        answer_content = ""
        is_answering = False
        print(return_audio, enable_thinking)
        for chunk in completion:
            if chunk.choices:
                if hasattr(chunk.choices[0].delta, "audio"):
                    try:
                        audio_string += chunk.choices[0].delta.audio["data"]
                    except Exception:
                        output_text += chunk.choices[0].delta.audio[
                            "transcript"]
                        yield {"type": "text", "data": output_text}
                else:
                    delta = chunk.choices[0].delta
                    if enable_thinking:
                        if hasattr(delta, "reasoning_content"
                                   ) and delta.reasoning_content is not None:
                            if not is_answering:
                                print(delta.reasoning_content,
                                      end="",
                                      flush=True)
                            reasoning_content += delta.reasoning_content
                            yield {"type": "text", "data": reasoning_content}
                        if hasattr(delta, "content") and delta.content:
                            if not is_answering:
                                reasoning_content += "\n\n</think>\n\n"
                                is_answering = True
                            answer_content += delta.content
                            yield {
                                "type": "text",
                                "data": reasoning_content + answer_content
                            }
                    else:
                        if hasattr(delta, "content") and delta.content:
                            output_text += chunk.choices[0].delta.content
                            yield {"type": "text", "data": output_text}
            else:
                print(chunk.usage)

        if audio_string:
            wav_bytes = base64.b64decode(audio_string)
            audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_np, samplerate=24000, format="WAV")
            wav_io.seek(0)
            wav_bytes = wav_io.getvalue()
            audio_path = processing_utils.save_bytes_to_cache(
                wav_bytes, "audio.wav", cache_dir=demo.GRADIO_CACHE)
            yield {"type": "audio", "data": audio_path}

    def media_predict(audio,
                      video,
                      history,
                      system_prompt,
                      voice_choice,
                      temperature,
                      top_p,
                      top_k,
                      return_audio=False,
                      enable_thinking=False):
        yield (
            None,
            None,
            history,
            gr.update(visible=False),
            gr.update(visible=True),
        )

        files = [audio, video]
        for f in files:
            if f:
                history.append({"role": "user", "content": (f,)})

        yield (
            None,
            None,
            history,
            gr.update(visible=True),
            gr.update(visible=False),
        )

        formatted_history = format_history(
            history=history,
            system_prompt=system_prompt,
        )

        history.append({"role": "assistant", "content": ""})

        for chunk in predict(formatted_history, voice_choice, temperature,
                             top_p, top_k, return_audio, enable_thinking):
            print('chunk', chunk)
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (
                    None,
                    None,
                    history,
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })

        yield (
            None,
            None,
            history,
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def chat_predict(text,
                     audio,
                     image,
                     video,
                     history,
                     system_prompt,
                     voice_choice,
                     temperature,
                     top_p,
                     top_k,
                     return_audio=False,
                     enable_thinking=False):
        if audio:
            history.append({"role": "user", "content": (audio,)})
        if text:
            history.append({"role": "user", "content": text})
        if image:
            history.append({"role": "user", "content": (image,)})
        if video:
            history.append({"role": "user", "content": (video,)})

        formatted_history = format_history(history=history,
                                           system_prompt=system_prompt)

        yield None, None, None, None, history

        history.append({"role": "assistant", "content": ""})
        for chunk in predict(formatted_history, voice_choice, temperature,
                             top_p, top_k, return_audio, enable_thinking):
            print('chat_predict chunk', chunk)

            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history

    # --- UI LAYOUT ---
    with gr.Blocks(
            theme=gr.themes.Soft(font=[
                gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"
            ]),
            css=".gradio-container {max-width: none !important;}") as demo:
        gr.Markdown("# Qwen3-Omni Demo (Local)")
        gr.Markdown(
            f"**Endpoint**: `{LOCAL_API_BASE}` | **Model**: `{LOCAL_MODEL_NAME}`"
        )
        gr.Markdown(
            "**Instructions**: Interact with the model through text, audio, images, or video. "
            "Use the tabs to switch between Online and Offline chat modes."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("### Parameters")
                system_prompt_textbox = gr.Textbox(label="System Prompt",
                                                   value=default_system_prompt,
                                                   lines=4,
                                                   max_lines=8)
                voice_choice = gr.Dropdown(label="Voice Choice",
                                           choices=VOICE_OPTIONS,
                                           value=DEFAULT_VOICE,
                                           visible=True)
                return_audio = gr.Checkbox(label="Return Audio",
                                           value=True,
                                           interactive=True,
                                           elem_classes="checkbox-large")
                enable_thinking = gr.Checkbox(label="Enable Thinking",
                                              value=False,
                                              interactive=True,
                                              elem_classes="checkbox-large")
                temperature = gr.Slider(label="Temperature",
                                        minimum=0.1,
                                        maximum=2.0,
                                        value=0.6,
                                        step=0.1)
                top_p = gr.Slider(label="Top P",
                                  minimum=0.05,
                                  maximum=1.0,
                                  value=0.95,
                                  step=0.05)
                top_k = gr.Slider(label="Top K",
                                  minimum=1,
                                  maximum=100,
                                  value=20,
                                  step=1)

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("Online"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Audio-Video Input")
                                microphone = gr.Audio(
                                    sources=['microphone'],
                                    type="filepath",
                                    label="Record Audio")
                                webcam = gr.Video(
                                    sources=['webcam', "upload"],
                                    label="Record/Upload Video",
                                    elem_classes="media-upload")
                                with gr.Row():
                                    submit_btn_online = gr.Button(
                                        "Submit",
                                        variant="primary",
                                        scale=2)
                                    stop_btn_online = gr.Button("Stop",
                                                                visible=False,
                                                                scale=1)
                                clear_btn_online = gr.Button("Clear History")
                            with gr.Column(scale=2):
                                media_chatbot = gr.Chatbot(
                                    label="Chat History",
                                    type="messages",
                                    height=650,
                                    layout="panel",
                                    bubble_full_width=False,
                                    allow_tags=["think"],
                                    render=False)
                                media_chatbot.render()

                        def clear_history_online():
                            return [], None, None

                        submit_event_online = submit_btn_online.click(
                            fn=media_predict,
                            inputs=[
                                microphone, webcam, media_chatbot,
                                system_prompt_textbox, voice_choice,
                                temperature, top_p, top_k, return_audio,
                                enable_thinking
                            ],
                            outputs=[
                                microphone, webcam, media_chatbot,
                                submit_btn_online, stop_btn_online
                            ])
                        stop_btn_online.click(
                            fn=lambda: (gr.update(visible=True),
                                        gr.update(visible=False)),
                            outputs=[submit_btn_online, stop_btn_online],
                            cancels=[submit_event_online],
                            queue=False)
                        clear_btn_online.click(
                            fn=clear_history_online,
                            outputs=[media_chatbot, microphone, webcam])

                    with gr.TabItem("Offline"):
                        chatbot = gr.Chatbot(label="Chat History",
                                             type="messages",
                                             height=550,
                                             layout="panel",
                                             bubble_full_width=False,
                                             allow_tags=["think"],
                                             render=False)
                        chatbot.render()

                        with gr.Accordion(
                                "Click to upload multimodal files",
                                open=False):
                            with gr.Row():
                                audio_input = gr.Audio(
                                    sources=["upload", 'microphone'],
                                    type="filepath",
                                    label="Audio",
                                    elem_classes="media-upload")
                                image_input = gr.Image(
                                    sources=["upload", 'webcam'],
                                    type="filepath",
                                    label="Image",
                                    elem_classes="media-upload")
                                video_input = gr.Video(
                                    sources=["upload", 'webcam'],
                                    label="Video",
                                    elem_classes="media-upload")

                        with gr.Row():
                            text_input = gr.Textbox(
                                show_label=False,
                                placeholder="Enter text or upload files and press Submit...",
                                scale=7)
                            submit_btn_offline = gr.Button("Submit",
                                                           variant="primary",
                                                           scale=1)
                            stop_btn_offline = gr.Button("Stop",
                                                         visible=False,
                                                         scale=1)
                            clear_btn_offline = gr.Button("Clear",
                                                          scale=1)

                        def clear_history_offline():
                            return [], None, None, None, None

                        submit_event_offline = gr.on(
                            triggers=[
                                submit_btn_offline.click, text_input.submit
                            ],
                            fn=chat_predict,
                            inputs=[
                                text_input, audio_input, image_input,
                                video_input, chatbot, system_prompt_textbox,
                                voice_choice, temperature, top_p, top_k,
                                return_audio, enable_thinking
                            ],
                            outputs=[
                                text_input, audio_input, image_input,
                                video_input, chatbot
                            ])
                        stop_btn_offline.click(
                            fn=lambda: (gr.update(visible=True),
                                        gr.update(visible=False)),
                            outputs=[submit_btn_offline, stop_btn_offline],
                            cancels=[submit_event_offline],
                            queue=False)
                        clear_btn_offline.click(fn=clear_history_offline,
                                                outputs=[
                                                    chatbot, text_input,
                                                    audio_input, image_input,
                                                    video_input
                                                ])

        gr.HTML("""
            <style>
                .media-upload { min-height: 160px; border: 2px dashed #ccc; border-radius: 8px; display: flex; align-items: center; justify-content: center; }
                .media-upload:hover { border-color: #666; }
            </style>
        """)

    demo.queue(default_concurrency_limit=100, max_size=100).launch(
        max_threads=100,
        ssr_mode=False,
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


DEFAULT_CKPT_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only',
                        action='store_true',
                        help='Run demo with CPU only')
    parser.add_argument(
        '--flash-attn2',
        action='store_true',
        default=False,
        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--use-transformers',
                        action='store_true',
                        default=False,
                        help='Use transformers for inference.')
    parser.add_argument(
        '--share',
        action='store_true',
        default=False,
        help='Create a publicly shareable link for the interface.')
    parser.add_argument(
        '--inbrowser',
        action='store_true',
        default=False,
        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port',
                        type=int,
                        default=7860,
                        help='Demo server port.')
    parser.add_argument('--server-name',
                        type=str,
                        default='0.0.0.0',
                        help='Demo server name.')
    parser.add_argument('--ui-language',
                        type=str,
                        choices=['en', 'zh'],
                        default='en',
                        help='Display language for the UI.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)
