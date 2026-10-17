from JpegImage import JpegImage
import requests
import json
from ResponseParser import LLMResponseParser
import LLMBackend

class LLM:
    def __init__(self, api_url: str, model: str = "local-model", timeout: float = 300):
        self.api_url = api_url
        self.model = model or "local-model"
        self.timeout = timeout

    def call_llm(self, image: JpegImage, prompt: str) -> str | None:
        headers = {"Content-Type": "application/json"}
        # Works for both OpenAI-compatible servers and Ollama (chosen by the URL, see LLMBackend)
        payload = LLMBackend.build_chat_payload(self.api_url, self.model, prompt, image.get_as_base64())

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            content = LLMBackend.extract_reply(self.api_url, response.json())
            # Clean any noisy wrappers like [THINK]...[/THINK] or code fences
            return LLMResponseParser.clean_content(content)
        except requests.exceptions.RequestException as e:
            detail = ""
            resp = getattr(e, "response", None)
            if resp is not None:
                detail = f" | Server says: {LLMBackend.error_detail(self.api_url, resp)}"
            print(f"API request failed: {e}{detail}")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            print(f"Could not parse API response: {e}")
        return None
