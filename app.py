import os
import logging
from jinja2 import Environment, FileSystemLoader
from time import perf_counter

import gradio as gr

from backend.query_llm import generate_openai, generate_hf
from backend.semantic_search import retrieve
from backend.embedder import embedder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("TOP_K", 4))

env = Environment(loader=FileSystemLoader("./templates"))

template = env.get_template("template.j2")
template_html = env.get_template("template_html.j2")


def add_text(history, text):
    history = [] if history is None else history
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def bot(history, vs_name, api_kind, top_k: int = 3):
    query = history[-1][0]

    if not query:
        raise gr.Warning("Please submit a non-empty string as a prompt")

    logger.info("Retrieving documents...")

    doc_start = perf_counter()
    doc = retrieve(vs_name, query, k=25, rerank=True, top_k=top_k)
    doc_time = perf_counter() - doc_start
    logger.info(
        f"Finished Retrieving documents in \
        {round(doc_time, 2)} seconds..."
    )

    # Create Prompt
    prompt = template.render(documents=doc, query=query, history=history)
    prompt_html = template_html.render(documents=doc, query=query, history=history)

    if api_kind == "HuggingFace":
        generate_fn = generate_hf
    elif api_kind == "OpenAI":
        generate_fn = generate_openai
    else:
        raise gr.Error(f"API {api_kind} is not supported")

    history[-1] = (history[-1][0], "")

    for character in generate_fn(prompt):
        history[-1] = (history[-1][0], character)
        yield history, prompt_html


def var_textbox(x):
    return x


with gr.Blocks() as demo:

    vs_name_state = gr.State()

    with gr.Row():
        file_input = gr.File(type="filepath")
        upload_btn = gr.Button(value="Upload file")

    vs_name_output = gr.Textbox(label="Vector Store Name")

    upload_btn.click(embedder, inputs=file_input, outputs=vs_name_output).then(
        var_textbox, inputs=vs_name_output, outputs=vs_name_state
    )

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        avatar_images=(
            "https://aui.atlassian.com/aui/8.8/docs/images/avatar-person.svg",
            "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg",
        ),
        bubble_full_width=False,
        show_copy_button=True,
        show_share_button=True,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=3,
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
        )
        txt_btn = gr.Button(value="Submit text", scale=1)

    api_kind_option = gr.Radio(choices=["HuggingFace", "OpenAI"], value="HuggingFace")
    api_topk_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Top-K")

    prompt_html = gr.HTML()
    # Turn off interactivity while generating if you click
    txt_msg = txt_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot,
        [chatbot, vs_name_state, api_kind_option, api_topk_slider],
        [chatbot, prompt_html],
    )

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    # Turn off interactivity while generating if you hit enter
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot,
        [chatbot, vs_name_state, api_kind_option, api_topk_slider],
        [chatbot, prompt_html],
    )

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)


demo.queue()
demo.launch(debug=True)
