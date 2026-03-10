"""Audio composition: combine TTS segments into a single audio track."""

import os
import subprocess
import numpy as np

from rich.console import Console

console = Console()


def compose_audio(segments: list, total_duration: float, temp_dir: str, sample_rate: int = 44100) -> str:
    """Compose all TTS audio segments into a single WAV file with correct timing."""
    import soundfile as sf

    output_path = os.path.join(temp_dir, "translated_audio.wav")

    # Create silent audio buffer for the full duration
    total_samples = int(total_duration * sample_rate)
    audio_buffer = np.zeros(total_samples, dtype=np.float32)

    for seg in segments:
        if not seg.tts_audio or not os.path.isfile(seg.tts_audio):
            continue

        try:
            seg_audio, seg_sr = sf.read(seg.tts_audio, dtype="float32")

            # Resample if needed
            if seg_sr != sample_rate:
                seg_audio = _resample(seg_audio, seg_sr, sample_rate)

            # Mono
            if seg_audio.ndim > 1:
                seg_audio = seg_audio.mean(axis=1)

            # Calculate position
            start_sample = int(seg.start * sample_rate)
            end_sample = start_sample + len(seg_audio)

            # Clip to buffer bounds
            if end_sample > total_samples:
                seg_audio = seg_audio[:total_samples - start_sample]
                end_sample = total_samples

            if start_sample < total_samples:
                audio_buffer[start_sample:start_sample + len(seg_audio)] = seg_audio

        except Exception as e:
            console.print(f"[red]Failed to compose segment at {seg.start:.1f}s:[/red] {e}")

    sf.write(output_path, audio_buffer, sample_rate)
    console.print(f"[green]Composed translated audio:[/green] {output_path}")
    return output_path


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple resampling using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
