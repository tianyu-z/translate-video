"""Audio extraction and processing using ffmpeg."""

import os
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()


def _ensure_ffmpeg_path():
    """Add static-ffmpeg to PATH if system ffmpeg is not available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            import static_ffmpeg
            ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            if ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
                console.print(f"[dim]Added static-ffmpeg to PATH: {ffmpeg_dir}[/dim]")
        except Exception:
            pass


_ensure_ffmpeg_path()


def extract_audio(video_path: str, temp_dir: str) -> str:
    """Extract audio track from video as WAV (16kHz mono for Whisper)."""
    audio_path = os.path.join(temp_dir, "original_audio.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz for Whisper
        "-ac", "1",               # mono
        audio_path,
    ]
    console.print("[yellow]Extracting audio...[/yellow]")
    subprocess.run(cmd, check=True, capture_output=True)
    console.print(f"[green]Audio extracted:[/green] {audio_path}")
    return audio_path


def extract_audio_full_quality(video_path: str, temp_dir: str) -> str:
    """Extract original quality audio for voice cloning reference."""
    audio_path = os.path.join(temp_dir, "original_audio_hq.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        audio_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def extract_audio_segment(audio_path: str, start: float, end: float, output_path: str) -> str:
    """Extract a segment of audio given start/end times."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def adjust_audio_speed(audio_path: str, target_duration: float, output_path: str,
                       min_speed: float = 0.8, max_speed: float = 1.5) -> str:
    """Adjust audio speed to fit target duration using ffmpeg atempo filter."""
    import soundfile as sf

    data, sr = sf.read(audio_path)
    current_duration = len(data) / sr

    if current_duration <= 0:
        return audio_path

    speed_factor = current_duration / target_duration
    speed_factor = max(min_speed, min(max_speed, speed_factor))

    # atempo filter range is [0.5, 100], chain multiple for extremes
    filters = []
    remaining = speed_factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-filter:a", ",".join(filters),
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def merge_audio_to_video(
    video_path: str,
    translated_audio_path: str,
    output_path: str,
    keep_original: bool = True,
    original_volume: float = 0.1,
) -> str:
    """Replace video audio with translated audio, optionally mixing original at low volume.

    Uses adelay+amix with normalize=0 to prevent volume reduction.
    The translated audio plays at full volume; original is a quiet background.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if keep_original and original_volume > 0:
        # Mix without normalization: translated at full volume, original as background
        # normalize=0 prevents amix from dividing each input by number of inputs
        # dropout_transition=0 prevents fade-out when one stream is silent
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", translated_audio_path,
            "-filter_complex",
            f"[0:a]volume={original_volume}[bg];"
            f"[1:a]aresample=44100,volume=1.0[fg];"
            f"[bg][fg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]",
            "-map", "0:v",
            "-map", "[out]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path,
        ]
    else:
        # Replace audio entirely
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", translated_audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path,
        ]

    console.print("[yellow]Merging translated audio into video...[/yellow]")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]ffmpeg error:[/red] {result.stderr[-500:]}")
        raise RuntimeError(f"ffmpeg merge failed: {result.stderr[-200:]}")
    console.print(f"[green]Output video:[/green] {output_path}")
    return output_path
