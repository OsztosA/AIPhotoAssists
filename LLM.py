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

    def call_llm(self, image: JpegImage, prompt: str, max_dimension: int = 1024,
                 response_schema: dict | None = None, temperature: float | None = None,
                 schema_name: str = "response") -> str | None:
        """
        Sends the image + prompt. With response_schema the reply is grammar-constrained to that JSON schema; if the
        server rejects the schema (HTTP 400) the request is retried once without it.
        """
        headers = {"Content-Type": "application/json"}
        image_b64 = image.get_as_base64(max_dimension)

        def build(schema: dict | None) -> dict:
            # Works for both OpenAI-compatible servers and Ollama (chosen by the URL, see LLMBackend)
            return LLMBackend.build_chat_payload(self.api_url, self.model, prompt, image_b64,
                                                 temperature=temperature, response_schema=schema,
                                                 schema_name=schema_name)

        try:
            response = requests.post(self.api_url, headers=headers, json=build(response_schema),
                                     timeout=self.timeout)
            if response.status_code == 400 and response_schema is not None:
                print("Server rejected the response schema (HTTP 400: "
                      f"{LLMBackend.error_detail(self.api_url, response)}); retrying without constrained decoding.")
                response = requests.post(self.api_url, headers=headers, json=build(None), timeout=self.timeout)
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
