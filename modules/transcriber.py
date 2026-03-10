"""Speech-to-text transcription using faster-whisper, or subtitle parsing."""

import os
import re
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.progress import Progress

console = Console()


@dataclass
class Segment:
    """A transcribed segment with timing information."""
    start: float       # seconds
    end: float         # seconds
    text: str          # original transcribed text
    translated: str = ""  # translated text
    tts_audio: str = ""   # path to TTS audio file


# ── Subtitle file parsing ──

def parse_subtitle_file(sub_path: str) -> tuple[list[Segment], str | None]:
    """Parse SRT/VTT subtitle file into segments.

    Returns:
        (segments, detected_language or None)
    """
    ext = os.path.splitext(sub_path)[1].lower()
    with open(sub_path, "r", encoding="utf-8") as f:
        content = f.read()

    if ext == ".srt":
        segments = _parse_srt(content)
    elif ext == ".vtt":
        segments = _parse_vtt(content)
    elif ext == ".ass":
        segments = _parse_ass(content)
    else:
        raise ValueError(f"Unsupported subtitle format: {ext}")

    # Try to detect language from filename (e.g. input.en.srt)
    lang = _detect_lang_from_filename(sub_path)

    console.print(f"[green]Parsed subtitle:[/green] {len(segments)} segments from {os.path.basename(sub_path)}")
    if lang:
        console.print(f"[cyan]Subtitle language:[/cyan] {lang}")

    return segments, lang


def _parse_srt(content: str) -> list[Segment]:
    """Parse SRT format."""
    segments = []
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Find the timestamp line
        ts_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                ts_line = line
            elif ts_line is not None:
                text_lines.append(line.strip())

        if not ts_line or not text_lines:
            continue

        start, end = _parse_srt_timestamp_line(ts_line)
        text = " ".join(text_lines)
        # Strip HTML tags (some SRT files have <i>, <b> etc.)
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            segments.append(Segment(start=start, end=end, text=text))

    return segments


def _parse_vtt(content: str) -> list[Segment]:
    """Parse WebVTT format."""
    segments = []
    # Remove WEBVTT header and metadata
    content = re.sub(r"^WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
    # Remove NOTE blocks
    content = re.sub(r"NOTE\s.*?\n\n", "", content, flags=re.DOTALL)

    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        ts_line = None
        text_lines = []

        for line in lines:
            if "-->" in line:
                ts_line = line
            elif ts_line is not None:
                text_lines.append(line.strip())

        if not ts_line or not text_lines:
            continue

        start, end = _parse_vtt_timestamp_line(ts_line)
        text = " ".join(text_lines)
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            segments.append(Segment(start=start, end=end, text=text))

    return segments


def _parse_ass(content: str) -> list[Segment]:
    """Parse ASS/SSA subtitle format."""
    segments = []
    in_events = False

    for line in content.split("\n"):
        line = line.strip()
        if line == "[Events]":
            in_events = True
            continue
        if line.startswith("[") and in_events:
            break
        if in_events and line.startswith("Dialogue:"):
            parts = line.split(",", 9)  # ASS has 10 fields
            if len(parts) >= 10:
                start = _parse_ass_time(parts[1].strip())
                end = _parse_ass_time(parts[2].strip())
                text = parts[9].strip()
                # Remove ASS override tags like {\pos(x,y)}
                text = re.sub(r"\{[^}]*\}", "", text)
                # Replace \N with space
                text = text.replace("\\N", " ").replace("\\n", " ").strip()
                if text:
                    segments.append(Segment(start=start, end=end, text=text))

    return segments


def _parse_srt_timestamp_line(line: str) -> tuple[float, float]:
    """Parse '00:01:23,456 --> 00:01:25,789'"""
    match = re.search(
        r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})",
        line,
    )
    if not match:
        return 0.0, 0.0
    return _ts_to_seconds(match.group(1)), _ts_to_seconds(match.group(2))


def _parse_vtt_timestamp_line(line: str) -> tuple[float, float]:
    """Parse VTT timestamps — same as SRT but may omit hours."""
    # Remove position/alignment metadata after timestamp
    line = re.sub(r"\s+(position|align|size|line):.*$", "", line)
    match = re.search(
        r"(\d{1,2}:?\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:?\d{2}:\d{2}[,\.]\d{3})",
        line,
    )
    if not match:
        return 0.0, 0.0
    return _ts_to_seconds(match.group(1)), _ts_to_seconds(match.group(2))


def _ts_to_seconds(ts: str) -> float:
    """Convert 'HH:MM:SS,mmm' or 'MM:SS.mmm' to seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


def _parse_ass_time(ts: str) -> float:
    """Parse ASS timestamp 'H:MM:SS.CC' (centiseconds)."""
    match = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d{2})", ts)
    if not match:
        return 0.0
    h, m, s, cs = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100


def _detect_lang_from_filename(sub_path: str) -> str | None:
    """Try to extract language code from subtitle filename.

    e.g. 'input.en.srt' -> 'en', 'input.zh-Hans.vtt' -> 'zh'
    """
    stem = os.path.splitext(os.path.basename(sub_path))[0]  # 'input.en'
    parts = stem.split(".")
    if len(parts) >= 2:
        lang_part = parts[-1].replace("-auto", "")
        # Normalize: 'zh-Hans' -> 'zh', 'pt-BR' -> 'pt'
        lang_code = lang_part.split("-")[0].lower()
        if len(lang_code) in (2, 3):
            return lang_code
    return None


# ── Whisper transcription ──

def transcribe(audio_path: str, config: dict, source_lang: str = "auto") -> list[Segment]:
    """Transcribe audio using faster-whisper. Returns list of timed segments."""
    from faster_whisper import WhisperModel

    tc = config.get("transcription", {})
    model_name = tc.get("model", "large-v3")
    device = tc.get("device", "auto")
    compute_type = tc.get("compute_type", "float16")

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            compute_type = "int8"

    console.print(f"[yellow]Loading Whisper model:[/yellow] {model_name} ({device}, {compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    lang = None if source_lang == "auto" else source_lang

    console.print("[yellow]Transcribing audio...[/yellow]")
    segments_iter, info = model.transcribe(
        audio_path,
        language=lang,
        beam_size=tc.get("beam_size", 5),
        vad_filter=tc.get("vad_filter", True),
        word_timestamps=tc.get("word_timestamps", True),
    )

    detected_lang = info.language
    console.print(f"[cyan]Detected language:[/cyan] {detected_lang} (prob: {info.language_probability:.2f})")

    segments = []
    for seg in segments_iter:
        s = Segment(start=seg.start, end=seg.end, text=seg.text.strip())
        if s.text:
            segments.append(s)

    console.print(f"[green]Transcription complete:[/green] {len(segments)} segments")
    return segments, detected_lang


def segments_to_srt(segments: list[Segment], use_translated: bool = False) -> str:
    """Convert segments to SRT subtitle format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg.translated if use_translated and seg.translated else seg.text
        start_ts = _format_srt_time(seg.start)
        end_ts = _format_srt_time(seg.end)
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
