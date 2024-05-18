import os
from typing import Any
from typing import Dict
from typing import Generator
from typing import List

import gradio as gr
import openai
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer


OPENAI_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("HF_MODEL"))

HF_CLIENT = InferenceClient(os.getenv("HF_MODEL"), token=HF_TOKEN)
OAI_CLIENT = openai.Client(api_key=OPENAI_KEY)

HF_GENERATE_KWARGS = {
    "temperature": max(float(os.getenv("TEMPERATURE", 0.9)), 1e-2),
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", 256)),
    "top_p": float(os.getenv("TOP_P", 0.6)),
    "repetition_penalty": float(os.getenv("REP_PENALTY", 1.2)),
    "do_sample": bool(os.getenv("DO_SAMPLE", True)),
}

OAI_GENERATE_KWARGS = {
    "temperature": max(float(os.getenv("TEMPERATURE", 0.9)), 1e-2),
    "max_tokens": int(os.getenv("MAX_NEW_TOKENS", 256)),
    "top_p": float(os.getenv("TOP_P", 0.6)),
    "frequency_penalty": max(-2, min(float(os.getenv("FREQ_PENALTY", 0)), 2)),
}


def format_prompt(message: str, api_kind: str):
    """
    Formats the given message using a chat template.

    Args:
        message (str): The user message to be formatted.
        api_kind (str): LLM API provider.
    Returns:
        str: Formatted message after applying the chat template.
    """

    messages: List[Dict[str, Any]] = [{"role": "user", "content": message}]

    if api_kind == "openai":
        return messages
    elif api_kind == "hf":
        return TOKENIZER.apply_chat_template(messages, tokenize=False)
    else:
        raise ValueError("API is not supported")


def generate_hf(prompt: str) -> Generator[str, None, str]:
    """
    Generate a sequence of tokens based on a given prompt and history using Mistral client.

    Args:
        prompt (str): The prompt for the text generation.
        history (str): Context or history for the text generation.
    Returns:
        Generator[str, None, str]: A generator yielding chunks of generated text.
                                   Returns a final string if an error occurs.
    """

    formatted_prompt = format_prompt(prompt, "hf")
    formatted_prompt = formatted_prompt.encode("utf-8").decode("utf-8")

    try:
        stream = HF_CLIENT.text_generation(
            formatted_prompt,
            **HF_GENERATE_KWARGS,
            stream=True,
            details=True,
            return_full_text=False,
        )
        output = ""
        for response in stream:
            output += response.token.text
            yield output

    except Exception as e:
        if "Too Many Requests" in str(e):
            raise gr.Error(f"Too many requests: {str(e)}")
        elif "Authorization header is invalid" in str(e):
            raise gr.Error(
                "Authentication error: HF token was either not provided or incorrect"
            )
        else:
            raise gr.Error(f"Unhandled Exception: {str(e)}")


def generate_openai(prompt: str) -> Generator[str, None, str]:
    """
    Generate a sequence of tokens based on a given prompt and history using OPENAI client.

    Args:
        prompt (str): The initial prompt for the text generation.
        history (str): Context or history for the text generation.
    Returns:
        Generator[str, None, str]: A generator yielding chunks of generated text.
                                   Returns a final string if an error occurs.
    """
    formatted_prompt = format_prompt(prompt, "openai")

    try:
        stream = OAI_CLIENT.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=formatted_prompt,
            **OAI_GENERATE_KWARGS,
            stream=True,
        )
        output = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                output += chunk.choices[0].delta.content
                yield output

    except Exception as e:
        if "Too Many Requests" in str(e):
            raise gr.Error("ERROR: Too many requests on OpenAI client")
        elif "You didn't provide an API key" in str(e):
            raise gr.Error(
                "Authentication error: OpenAI key was either not provided or incorrect"
            )
        else:
            raise gr.Error(f"Unhandled Exception: {str(e)}")
