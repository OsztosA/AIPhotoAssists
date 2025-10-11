# Utility for cleaning and parsing LLM responses
import re
import json
from typing import Optional


class LLMResponseParser:
    @staticmethod
    def clean_content(text: Optional[str]) -> str:
        """
        Cleans common noisy wrappers from LLM responses:
        - Removes [THINK]...[/THINK] or [THINK]...[\THINK] blocks (case-insensitive)
        - Strips markdown code fences like ```json ... ```
        - Trims whitespace
        Returns a best-effort cleaned string; if input is None, returns empty string.
        """
        if not isinstance(text, str):
            return ""

        cleaned = text
        # Remove [THINK]...[/THINK] or [THINK]...[\THINK] blocks (support multiple occurrences)
        think_pattern = re.compile(r"\[\s*THINK\s*\].*?\[(?:\s*/|\\)\s*THINK\s*\]", re.IGNORECASE | re.DOTALL)
        cleaned = think_pattern.sub("", cleaned)

        # Remove code fences like ```json ... ``` or ``` ... ```
        cleaned = re.sub(r"```[a-zA-Z]*\n?", "", cleaned)
        cleaned = cleaned.replace("```", "")

        # Collapse excessive whitespace and trim
        cleaned = cleaned.strip()
        return cleaned

    @staticmethod
    def extract_first_int_in_range(text: Optional[str], min_value: int = 0, max_value: int = 100) -> Optional[int]:
        """Extracts the first integer in text and validates it's within [min_value, max_value]."""
        if not isinstance(text, str):
            return None
        match = re.search(r"\d+", text)
        if not match:
            return None
        try:
            value = int(match.group(0))
            if min_value <= value <= max_value:
                return value
        except ValueError:
            return None
        return None

    @staticmethod
    def extract_json_object(text: Optional[str]) -> Optional[dict]:
        """
        Attempts to parse a JSON object from text. Tries direct json.loads first,
        then searches for a {...} block (greedy then non-greedy) and parses it.
        Returns the parsed dict or None if parsing fails.
        """
        if not isinstance(text, str) or not text:
            return None
        # Fast path
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # Try to find a JSON object greedily
        for pattern in (r"\{.*\}", r"\{.*?\}"):
            m = re.search(pattern, text, re.DOTALL)
            if m:
                snippet = m.group(0)
                try:
                    data = json.loads(snippet)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    continue
        return None
