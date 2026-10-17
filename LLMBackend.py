"""Supported LLM server flavours and how to talk to them.

The flavour is derived from the API URL, so a single URL string still fully describes a server:
  * OpenAI-compatible (LM Studio, llama.cpp, vLLM, OpenAI, ...): .../v1/chat/completions
  * Ollama (native API):                                          .../api/chat
"""
import re

BACKEND_OPENAI = "OpenAI-compatible"
BACKEND_OLLAMA = "Ollama"
BACKENDS = (BACKEND_OPENAI, BACKEND_OLLAMA)

DEFAULT_URLS = {
    BACKEND_OPENAI: "http://localhost:1234/v1/chat/completions",
    BACKEND_OLLAMA: "http://localhost:11434/api/chat",
}


def is_ollama_url(api_url: str) -> bool:
    return api_url.strip().rstrip("/").endswith("/api/chat")


def backend_for_url(api_url: str) -> str:
    return BACKEND_OLLAMA if is_ollama_url(api_url) else BACKEND_OPENAI


def ollama_base_url(api_url: str) -> str:
    """'http://host:11434/api/chat' -> 'http://host:11434'"""
    return re.sub(r"/api/chat$", "", api_url.strip().rstrip("/"))


def build_chat_payload(api_url: str, model: str, prompt: str, image_b64: str | None = None,
                       max_tokens: int | None = None, temperature: float | None = None) -> dict:
    """Builds the request body for a single user message (optionally with one JPEG image)."""
    if is_ollama_url(api_url):
        message: dict = {"role": "user", "content": prompt}
        if image_b64:
            message["images"] = [image_b64]  # Ollama takes raw base64, without a data: prefix
        payload: dict = {"model": model, "stream": False, "messages": [message]}
        options = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if options:
            payload["options"] = options
        return payload

    content: object = prompt
    if image_b64:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    payload = {"model": model, "messages": [{"role": "user", "content": content}]}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def extract_reply(api_url: str, data: dict) -> str:
    """Pulls the assistant text out of a chat response (raises KeyError/IndexError if malformed)."""
    if is_ollama_url(api_url):
        return data["message"]["content"]
    return data["choices"][0]["message"]["content"]


def error_detail(api_url: str, response) -> str:
    """Best-effort server error message from a failed response."""
    try:
        err = response.json()["error"]
        return err if isinstance(err, str) else err["message"]
    except (ValueError, KeyError, TypeError):
        return response.text[:300]
