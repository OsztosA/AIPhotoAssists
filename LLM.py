from JpegImage import JpegImage
import requests
import json
from ResponseParser import LLMResponseParser

class LLM:
    def __init__(self, api_url: str, model: str = "local-model", timeout: float = 300):
        self.api_url = api_url
        self.model = model or "local-model"
        self.timeout = timeout

    def call_llm(self, image: JpegImage, prompt: str) -> str | None:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,

            "messages": [
                {
                    "role": "user",
                    "content": [
                        {   "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image.get_as_base64()}"
                            }
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            # Clean any noisy wrappers like [THINK]...[/THINK] or code fences
            return LLMResponseParser.clean_content(content)
        except requests.exceptions.RequestException as e:
            detail = ""
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    detail = f" | Server says: {resp.json()['error']['message']}"
                except (ValueError, KeyError, TypeError):
                    detail = f" | Server says: {resp.text[:300]}"
            print(f"API request failed: {e}{detail}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Could not parse API response: {e}")
        return None
