"""Multi-backend translation module.

Supports: Ollama, vLLM, SGLang, HuggingFace, OpenAI API, Claude API.
"""

import os
from abc import ABC, abstractmethod

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Language code to natural name mapping
LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "ru": "Russian", "ar": "Arabic", "hi": "Hindi", "it": "Italian",
    "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "tr": "Turkish",
}


def get_lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


class TranslatorBackend(ABC):
    @abstractmethod
    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str, config: dict) -> list[str]:
        pass


class OllamaTranslator(TranslatorBackend):
    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str, config: dict) -> list[str]:
        import requests

        oc = config["translation"]["ollama"]
        base_url = oc["base_url"]
        model = oc["model"]
        prompt_tpl = config["translation"]["batch_prompt"]

        numbered_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = prompt_tpl.format(
            src_lang=get_lang_name(src_lang),
            tgt_lang=get_lang_name(tgt_lang),
            text=numbered_text,
        )

        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return _parse_batch_response(resp.json()["response"], len(texts))


class OpenAICompatibleTranslator(TranslatorBackend):
    """Works with vLLM, SGLang, and OpenAI API (all OpenAI-compatible)."""

    def __init__(self, backend_key: str):
        self.backend_key = backend_key

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str, config: dict) -> list[str]:
        from openai import OpenAI

        bc = config["translation"][self.backend_key]
        base_url = bc["base_url"]
        api_key = bc.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "no-key")
        model = bc["model"]

        client = OpenAI(base_url=base_url, api_key=api_key)

        prompt_tpl = config["translation"]["batch_prompt"]
        numbered_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = prompt_tpl.format(
            src_lang=get_lang_name(src_lang),
            tgt_lang=get_lang_name(tgt_lang),
            text=numbered_text,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return _parse_batch_response(response.choices[0].message.content, len(texts))


class ClaudeTranslator(TranslatorBackend):
    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str, config: dict) -> list[str]:
        import anthropic

        cc = config["translation"]["claude_api"]
        api_key = cc.get("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        model = cc.get("model", "claude-sonnet-4-20250514")

        client = anthropic.Anthropic(api_key=api_key)

        prompt_tpl = config["translation"]["batch_prompt"]
        numbered_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = prompt_tpl.format(
            src_lang=get_lang_name(src_lang),
            tgt_lang=get_lang_name(tgt_lang),
            text=numbered_text,
        )

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_batch_response(response.content[0].text, len(texts))


class HuggingFaceTranslator(TranslatorBackend):
    """HuggingFace Seq2Seq translation.

    Auto-selects Helsinki-NLP/opus-mt model based on src→tgt language direction.
    Can be overridden with explicit model name in config.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded_model_name = None

    def _pick_model(self, src_lang: str, tgt_lang: str, config: dict) -> str:
        """Auto-select translation model based on language direction."""
        hc = config["translation"]["huggingface"]
        explicit = hc.get("model", "")

        # If user set a specific model and it's not the default placeholder, use it
        if explicit and "opus-mt-en-zh" not in explicit:
            return explicit

        # Auto-select Helsinki-NLP model by language pair
        # Map common codes to Helsinki-NLP codes
        code_map = {
            "zh": "zh", "en": "en", "ja": "jap", "ko": "ko",
            "fr": "fr", "de": "de", "es": "es", "ru": "ru",
            "pt": "pt", "ar": "ar", "hi": "hi", "it": "it",
        }
        src = code_map.get(src_lang, src_lang)
        tgt = code_map.get(tgt_lang, tgt_lang)
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        console.print(f"[cyan]Auto-selected model:[/cyan] {model_name}")
        return model_name

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str, config: dict) -> list[str]:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = self._pick_model(src_lang, tgt_lang, config)

        # Reload if language direction changed
        if self._model is None or self._loaded_model_name != model_name:
            console.print(f"[yellow]Loading HuggingFace model:[/yellow] {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._loaded_model_name = model_name

        results = []
        for text in texts:
            inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self._model.generate(**inputs, max_length=512)
            translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(translated)
        return results


def _parse_batch_response(response: str, expected_count: int) -> list[str]:
    """Parse numbered batch translation response into list of strings."""
    import re

    lines = response.strip().split("\n")
    results = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove numbering like "1. ", "1) ", "1: "
        cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line)
        if cleaned:
            results.append(cleaned)

    # If parsing gave wrong count, fall back to splitting evenly
    if len(results) != expected_count:
        # Try to salvage: just take first N non-empty lines
        all_lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        all_lines = [re.sub(r"^\d+[\.\)\:]\s*", "", l) for l in all_lines]
        if len(all_lines) >= expected_count:
            results = all_lines[:expected_count]
        else:
            # Last resort: pad with originals or truncate
            results = results[:expected_count]
            while len(results) < expected_count:
                results.append("[translation failed]")

    return results


def get_translator(backend: str) -> TranslatorBackend:
    """Factory to get the right translator backend."""
    backends = {
        "ollama": OllamaTranslator,
        "vllm": lambda: OpenAICompatibleTranslator("vllm"),
        "sglang": lambda: OpenAICompatibleTranslator("sglang"),
        "openai_api": lambda: OpenAICompatibleTranslator("openai_api"),
        "claude_api": ClaudeTranslator,
        "huggingface": HuggingFaceTranslator,
    }
    factory = backends.get(backend)
    if not factory:
        raise ValueError(f"Unknown translation backend: {backend}. Choose from: {list(backends.keys())}")
    return factory() if callable(factory) else factory


def translate_segments(segments: list, config: dict, src_lang: str, tgt_lang: str) -> list:
    """Translate all segments using the configured backend."""
    backend_name = config["translation"]["backend"]
    batch_size = config["translation"].get("batch_size", 10)

    console.print(f"[yellow]Translating with:[/yellow] {backend_name}")
    console.print(f"[cyan]{get_lang_name(src_lang)} → {get_lang_name(tgt_lang)}[/cyan]")

    translator = get_translator(backend_name)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console) as progress:
        task = progress.add_task("Translating...", total=len(segments))

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            texts = [s.text for s in batch]

            translations = translator.translate_batch(texts, src_lang, tgt_lang, config)

            for seg, trans in zip(batch, translations):
                seg.translated = trans

            progress.advance(task, len(batch))

    console.print(f"[green]Translation complete:[/green] {len(segments)} segments")
    return segments
