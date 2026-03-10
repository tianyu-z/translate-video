"""Main pipeline: orchestrates the full video translation workflow.

Pipeline steps:
  1. Download/copy video (reuses cached files if available)
  2. Extract audio
  3. Transcribe (or use existing subtitles — skips Whisper)
  4. Translate text
  5. Synthesize speech (TTS with voice cloning)
  6. Compose translated audio track
  7. Merge into final video
"""

import os
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from modules.downloader import download_video
from modules.audio import (
    extract_audio,
    extract_audio_full_quality,
    extract_audio_segment,
    get_video_duration,
    merge_audio_to_video,
)
from modules.transcriber import transcribe, parse_subtitle_file, segments_to_srt
from modules.translator import translate_segments
from modules.tts import synthesize_segments
from modules.composer import compose_audio

console = Console()


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(
    source: str,
    config_path: str = "config.yaml",
    source_lang: str = None,
    target_lang: str = None,
    output_path: str = None,
    max_segments: int = 0,
) -> str:
    """Run the full video translation pipeline.

    Args:
        source: Local file path or URL (YouTube/Bilibili)
        config_path: Path to config.yaml
        source_lang: Override source language (default: from config or auto)
        target_lang: Override target language (default: from config)
        output_path: Override output file path
        max_segments: Limit number of segments (0 = all, useful for testing)

    Returns:
        Path to the translated video file.
    """
    config = load_config(config_path)

    src_lang = source_lang or config.get("source_language", "auto")
    tgt_lang = target_lang or config.get("target_language", "zh")
    temp_dir = os.path.abspath(config.get("temp_dir", "./temp"))
    out_dir = os.path.abspath(config.get("output_dir", "./output"))

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    start_time = time.time()

    console.print(Panel.fit(
        f"[bold]Video Translation Pipeline[/bold]\n"
        f"Source: {source}\n"
        f"Language: {src_lang} → {tgt_lang}\n"
        f"Translation: {config['translation']['backend']}\n"
        f"TTS: {config['tts']['backend']}"
        + (f"\nMax segments: {max_segments}" if max_segments > 0 else ""),
        title="Config",
    ))

    # ── Step 1: Download / Copy Video ──
    console.rule("[bold blue]Step 1: Get Video")

    # Reuse cached video if it exists in temp dir
    cached_video = _find_cached_video(temp_dir)
    cached_sub = _find_cached_subtitle(temp_dir)

    if cached_video:
        console.print(f"[green]Reusing cached video:[/green] {cached_video}")
        video_path = cached_video
        subtitle_path = cached_sub
    else:
        video_path, subtitle_path = download_video(source, config, temp_dir)

    # ── Step 2: Extract Audio ──
    console.rule("[bold blue]Step 2: Extract Audio")

    cached_audio = os.path.join(temp_dir, "original_audio.wav")
    if os.path.isfile(cached_audio):
        console.print(f"[green]Reusing cached audio[/green]")
        audio_path = cached_audio
    else:
        audio_path = extract_audio(video_path, temp_dir)

    duration = get_video_duration(video_path)
    console.print(f"[cyan]Video duration:[/cyan] {duration:.1f}s")

    # Extract reference audio for voice cloning (first N seconds of speaker)
    ref_audio_path = None
    tts_backend = config.get("tts", {}).get("backend", "edge_tts")
    if tts_backend in ("cosyvoice", "fish_speech"):
        hq_audio = extract_audio_full_quality(video_path, temp_dir)
        ref_duration = config.get("tts", {}).get(tts_backend, {}).get("ref_audio_duration", 10)
        ref_audio_path = os.path.join(temp_dir, "ref_audio.wav")
        extract_audio_segment(hq_audio, 0, min(ref_duration, duration), ref_audio_path)
        console.print(f"[cyan]Reference audio extracted:[/cyan] {ref_duration}s for voice cloning")

    # ── Step 3: Transcribe (or use existing subtitles) ──
    console.rule("[bold blue]Step 3: Transcribe")

    if subtitle_path:
        console.print(f"[green]Using existing subtitle file — skipping Whisper[/green]")
        segments, sub_lang = parse_subtitle_file(subtitle_path)
        detected_lang = sub_lang or src_lang
    else:
        console.print("[yellow]No subtitle found, running Whisper transcription...[/yellow]")
        segments, detected_lang = transcribe(audio_path, config, src_lang)

    if src_lang == "auto":
        src_lang = detected_lang if detected_lang else "en"

    # Limit segments for testing
    if max_segments > 0 and len(segments) > max_segments:
        console.print(f"[yellow]Limiting to first {max_segments} of {len(segments)} segments[/yellow]")
        segments = segments[:max_segments]

    console.print(f"[cyan]Segments to process:[/cyan] {len(segments)}")

    # Save original subtitles
    orig_srt = segments_to_srt(segments)
    orig_srt_path = os.path.join(temp_dir, "original.srt")
    with open(orig_srt_path, "w", encoding="utf-8") as f:
        f.write(orig_srt)

    # ── Step 4: Translate ──
    console.rule("[bold blue]Step 4: Translate")
    segments = translate_segments(segments, config, src_lang, tgt_lang)

    # Show sample
    table = Table(title="Translation Sample (first 5)")
    table.add_column("Time", style="cyan")
    table.add_column("Original", style="white")
    table.add_column("Translated", style="green")
    for seg in segments[:5]:
        table.add_row(
            f"{seg.start:.1f}-{seg.end:.1f}",
            seg.text[:50],
            seg.translated[:50],
        )
    console.print(table)

    # ── Step 5: TTS ──
    console.rule("[bold blue]Step 5: Text-to-Speech")

    # Detect speaker gender for voice matching
    from modules.tts import detect_speaker_gender
    speaker_gender = detect_speaker_gender(audio_path)

    segments = synthesize_segments(segments, config, tgt_lang, ref_audio_path, temp_dir, gender=speaker_gender)

    # ── Step 6: Compose Audio ──
    console.rule("[bold blue]Step 6: Compose Audio")
    translated_audio = compose_audio(segments, duration, temp_dir)

    # ── Step 7: Merge into Video ──
    console.rule("[bold blue]Step 7: Final Output")

    if not output_path:
        src_name = Path(source).stem if os.path.isfile(source) else "video"
        output_path = os.path.join(out_dir, f"{src_name}_translated_{tgt_lang}.mp4")

    out_config = config.get("output", {})
    final_video = merge_audio_to_video(
        video_path=video_path,
        translated_audio_path=translated_audio,
        output_path=output_path,
        keep_original=out_config.get("keep_original_audio", True),
        original_volume=out_config.get("original_audio_volume", 0.1),
    )

    # Save translated subtitles
    if out_config.get("generate_subtitles", True):
        sub_format = out_config.get("subtitle_format", "srt")
        sub_path = os.path.join(out_dir, f"{Path(output_path).stem}.{sub_format}")
        translated_srt = segments_to_srt(segments, use_translated=True)
        with open(sub_path, "w", encoding="utf-8") as f:
            f.write(translated_srt)
        console.print(f"[green]Subtitles saved:[/green] {sub_path}")

    elapsed = time.time() - start_time
    console.print(Panel.fit(
        f"[bold green]Done![/bold green]\n"
        f"Output: {final_video}\n"
        f"Time: {elapsed:.1f}s",
        title="Complete",
    ))

    return final_video


def _find_cached_video(temp_dir: str) -> str | None:
    """Find a previously downloaded video in temp dir."""
    for ext in (".mp4", ".mkv", ".webm"):
        path = os.path.join(temp_dir, f"input{ext}")
        if os.path.isfile(path):
            return path
    return None


def _find_cached_subtitle(temp_dir: str) -> str | None:
    """Find a previously downloaded subtitle in temp dir."""
    for f in Path(temp_dir).iterdir():
        if f.suffix in (".srt", ".vtt", ".ass") and f.name != "original.srt":
            return str(f)
    return None
