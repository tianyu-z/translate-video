"""Text-to-Speech module with voice cloning support.

Backends: CosyVoice2, Edge TTS, Fish Speech.
Includes speaker gender detection for automatic voice matching.
"""

import asyncio
import os
import tempfile
from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


# ── Speaker gender detection ──

def detect_speaker_gender(audio_path: str, duration: float = 10.0) -> str:
    """Detect speaker gender using wav2vec2-based audio classification model.

    Uses alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech,
    a fine-tuned model for gender recognition from speech audio.

    Args:
        audio_path: Path to audio file (16kHz mono WAV recommended).
        duration: Seconds of audio to analyze (default 10s).

    Returns: "male" or "female"
    """
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model_name = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
        console.print(f"[yellow]Loading gender detection model...[/yellow]")

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        model.eval()

        # Load audio segment
        data, sr = sf.read(audio_path)
        max_samples = int(duration * sr)
        if len(data) > max_samples:
            data = data[:max_samples]
        if data.ndim > 1:
            data = data.mean(axis=1)

        inputs = feature_extractor(data, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).item()

        gender = model.config.id2label[pred]
        confidence = probs[0][pred].item()

        console.print(
            f"[cyan]Speaker gender detection:[/cyan] "
            f"[bold]{'♀ female' if gender == 'female' else '♂ male'}[/bold] "
            f"(confidence: {confidence:.1%})"
        )
        return gender

    except Exception as e:
        console.print(f"[yellow]Gender detection failed ({e}), defaulting to male[/yellow]")
        return "male"


class TTSBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: str, ref_audio: str = None) -> str:
        """Synthesize speech. Returns path to output audio file."""
        pass

    def setup(self, config: dict, tts_config: dict):
        """Optional setup with config."""
        pass


class CosyVoiceTTS(TTSBackend):
    """CosyVoice2 voice cloning TTS via HTTP API.

    Requires running CosyVoice2 server separately.
    See: https://github.com/FunAudioLLM/CosyVoice
    """

    def __init__(self):
        self.base_url = ""
        self.ref_audio_path = None

    def setup(self, config: dict, tts_config: dict):
        cv_config = tts_config.get("cosyvoice", {})
        self.base_url = cv_config.get("base_url", "http://localhost:50000")

    def set_reference_audio(self, ref_audio_path: str):
        self.ref_audio_path = ref_audio_path

    def synthesize(self, text: str, output_path: str, ref_audio: str = None) -> str:
        import requests

        ref = ref_audio or self.ref_audio_path
        if not ref:
            raise ValueError("CosyVoice requires reference audio for voice cloning")

        # CosyVoice2 API - cross-lingual clone
        with open(ref, "rb") as f:
            files = {"ref_audio": ("ref.wav", f, "audio/wav")}
            data = {"text": text, "lang": "auto"}
            resp = requests.post(
                f"{self.base_url}/api/clone",
                files=files,
                data=data,
                timeout=60,
            )
            resp.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(resp.content)

        return output_path


class EdgeTTS(TTSBackend):
    """Microsoft Edge TTS - free, high quality, no cloning."""

    def __init__(self):
        self.voice = "zh-CN-YunxiNeural"

    def setup(self, config: dict, tts_config: dict):
        et_config = tts_config.get("edge_tts", {})
        self.voice = et_config.get("voice", "zh-CN-YunxiNeural")

    def synthesize(self, text: str, output_path: str, ref_audio: str = None) -> str:
        import edge_tts

        async def _synth():
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)

        asyncio.run(_synth())
        return output_path


class FishSpeechTTS(TTSBackend):
    """Fish Speech voice cloning TTS via HTTP API.

    See: https://github.com/fishaudio/fish-speech
    """

    def __init__(self):
        self.base_url = ""
        self.ref_audio_path = None

    def setup(self, config: dict, tts_config: dict):
        fs_config = tts_config.get("fish_speech", {})
        self.base_url = fs_config.get("base_url", "http://localhost:8080")

    def set_reference_audio(self, ref_audio_path: str):
        self.ref_audio_path = ref_audio_path

    def synthesize(self, text: str, output_path: str, ref_audio: str = None) -> str:
        import requests

        ref = ref_audio or self.ref_audio_path
        files = {}
        if ref:
            files["ref_audio"] = ("ref.wav", open(ref, "rb"), "audio/wav")

        data = {"text": text}
        resp = requests.post(
            f"{self.base_url}/v1/tts",
            files=files if files else None,
            data=data,
            timeout=60,
        )
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(resp.content)

        return output_path


# Voice mapping for Edge TTS by target language and gender
EDGE_VOICES = {
    "zh": {"male": "zh-CN-YunxiNeural",    "female": "zh-CN-XiaoxiaoNeural"},
    "en": {"male": "en-US-GuyNeural",       "female": "en-US-AriaNeural"},
    "ja": {"male": "ja-JP-KeitaNeural",     "female": "ja-JP-NanamiNeural"},
    "ko": {"male": "ko-KR-InJoonNeural",    "female": "ko-KR-SunHiNeural"},
    "fr": {"male": "fr-FR-HenriNeural",     "female": "fr-FR-DeniseNeural"},
    "de": {"male": "de-DE-ConradNeural",    "female": "de-DE-KatjaNeural"},
    "es": {"male": "es-ES-AlvaroNeural",    "female": "es-ES-ElviraNeural"},
    "pt": {"male": "pt-BR-AntonioNeural",   "female": "pt-BR-FranciscaNeural"},
    "ru": {"male": "ru-RU-DmitryNeural",    "female": "ru-RU-SvetlanaNeural"},
    "ar": {"male": "ar-SA-HamedNeural",     "female": "ar-SA-ZariyahNeural"},
    "hi": {"male": "hi-IN-MadhurNeural",    "female": "hi-IN-SwaraNeural"},
    "it": {"male": "it-IT-DiegoNeural",     "female": "it-IT-ElsaNeural"},
    "th": {"male": "th-TH-NiwatNeural",     "female": "th-TH-PremwadeeNeural"},
    "vi": {"male": "vi-VN-NamMinhNeural",   "female": "vi-VN-HoaiMyNeural"},
    "id": {"male": "id-ID-ArdiNeural",      "female": "id-ID-GadisNeural"},
    "tr": {"male": "tr-TR-AhmetNeural",     "female": "tr-TR-EmelNeural"},
}


def get_tts_backend(config: dict, tgt_lang: str, gender: str = "male") -> TTSBackend:
    """Factory to create TTS backend.

    Args:
        gender: "male" or "female" — used to select matching Edge TTS voice.
    """
    tts_config = config.get("tts", {})
    backend_name = tts_config.get("backend", "edge_tts")

    backends = {
        "cosyvoice": CosyVoiceTTS,
        "edge_tts": EdgeTTS,
        "fish_speech": FishSpeechTTS,
    }

    cls = backends.get(backend_name)
    if not cls:
        raise ValueError(f"Unknown TTS backend: {backend_name}. Choose from: {list(backends.keys())}")

    backend = cls()
    backend.setup(config, tts_config)

    # Auto-select edge voice by language and gender
    if backend_name == "edge_tts" and tgt_lang in EDGE_VOICES:
        voice_map = EDGE_VOICES[tgt_lang]
        backend.voice = voice_map.get(gender, voice_map["male"])
        console.print(f"[cyan]Selected TTS voice:[/cyan] {backend.voice} ({gender})")

    return backend


def synthesize_segments(
    segments: list,
    config: dict,
    tgt_lang: str,
    ref_audio_path: str = None,
    temp_dir: str = "./temp",
    gender: str = "male",
) -> list:
    """Generate TTS audio for all translated segments."""
    from modules.audio import adjust_audio_speed

    tts_config = config.get("tts", {})
    tts = get_tts_backend(config, tgt_lang, gender=gender)

    # Set reference audio for voice cloning backends
    if ref_audio_path and hasattr(tts, "set_reference_audio"):
        tts.set_reference_audio(ref_audio_path)

    speed_adj = tts_config.get("speed_adjustment", True)
    min_speed = tts_config.get("min_speed", 0.8)
    max_speed = tts_config.get("max_speed", 1.5)

    tts_dir = os.path.join(temp_dir, "tts_segments")
    os.makedirs(tts_dir, exist_ok=True)

    console.print(f"[yellow]Synthesizing speech ({tts_config.get('backend', 'edge_tts')})...[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("TTS...", total=len(segments))

        for i, seg in enumerate(segments):
            if not seg.translated:
                progress.advance(task)
                continue

            raw_path = os.path.join(tts_dir, f"seg_{i:04d}_raw.wav")
            final_path = os.path.join(tts_dir, f"seg_{i:04d}.wav")

            try:
                tts.synthesize(seg.translated, raw_path, ref_audio_path)

                if speed_adj:
                    target_duration = seg.end - seg.start
                    adjust_audio_speed(raw_path, target_duration, final_path, min_speed, max_speed)
                else:
                    final_path = raw_path

                seg.tts_audio = final_path
            except Exception as e:
                console.print(f"[red]TTS failed for segment {i}:[/red] {e}")
                seg.tts_audio = ""

            progress.advance(task)

    successful = sum(1 for s in segments if s.tts_audio)
    console.print(f"[green]TTS complete:[/green] {successful}/{len(segments)} segments")
    return segments
