"""Video downloader: supports local files, YouTube, and Bilibili.

Subtitle download strategy (priority):
  1. youtube-transcript-api (YouTube) / bilibili-api (Bilibili) — dedicated, reliable
  2. yt-dlp embedded subtitle download — fallback
  3. Sidecar subtitle files (local videos) — e.g. video.srt next to video.mp4
  4. None found → fall back to Whisper transcription in pipeline
"""

import glob
import json
import os
import re
import shutil
from pathlib import Path

from rich.console import Console

console = Console()


def is_url(source: str) -> bool:
    return bool(re.match(r"https?://", source))


def is_local_file(source: str) -> bool:
    return os.path.isfile(source)


def is_youtube_url(url: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)/", url))


def is_bilibili_url(url: str) -> bool:
    return bool(re.search(r"(bilibili\.com|b23\.tv)/", url))


def extract_youtube_video_id(url: str) -> str | None:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def extract_bilibili_bvid(url: str) -> str | None:
    """Extract BV ID from Bilibili URL."""
    match = re.search(r"(BV[a-zA-Z0-9]{10})", url)
    return match.group(1) if match else None


# ── Main entry point ──

def download_video(source: str, config: dict, temp_dir: str) -> tuple[str, str | None]:
    """Download or copy video to temp directory.

    Returns:
        (video_path, subtitle_path or None)
    """
    os.makedirs(temp_dir, exist_ok=True)

    if is_local_file(source):
        console.print(f"[green]Using local file:[/green] {source}")
        ext = Path(source).suffix
        dest = os.path.join(temp_dir, f"input{ext}")
        shutil.copy2(source, dest)
        sub_path = _find_sidecar_subtitle(source)
        return dest, sub_path

    if is_url(source):
        return _download_url(source, config, temp_dir)

    raise ValueError(f"Invalid source: {source}. Must be a file path or URL.")


# ── YouTube subtitle download via youtube-transcript-api ──

def _download_youtube_subtitles(url: str, temp_dir: str) -> str | None:
    """Download YouTube subtitles using youtube-transcript-api.

    Returns path to SRT file, or None if unavailable.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        console.print("[dim]youtube-transcript-api not installed, skipping[/dim]")
        return None

    video_id = extract_youtube_video_id(url)
    if not video_id:
        return None

    try:
        # v1.x API: instantiate then call .list()
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Priority: manual transcript > auto-generated
        transcript = None
        source_type = ""

        # Try manual transcripts first (en, en-US)
        try:
            transcript = transcript_list.find_manually_created_transcript(["en", "en-US"])
            source_type = "manual"
        except Exception:
            pass

        # Fall back to auto-generated
        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript(["en", "en-US"])
                source_type = "auto-generated"
            except Exception:
                pass

        # Last resort: any transcript
        if transcript is None:
            try:
                for t in transcript_list:
                    transcript = t
                    source_type = "auto" if t.is_generated else "manual"
                    break
            except Exception:
                pass

        if transcript is None:
            console.print("[yellow]No YouTube transcript available[/yellow]")
            return None

        lang = transcript.language_code
        console.print(f"[green]Found YouTube {source_type} transcript:[/green] {transcript.language} ({lang})")

        # Fetch the actual transcript data
        fetched = transcript.fetch()

        # Convert to SRT format — v1.x returns FetchedTranscript with .snippets
        snippets = fetched.snippets if hasattr(fetched, "snippets") else fetched
        srt_content = _transcript_entries_to_srt(snippets)
        srt_path = os.path.join(temp_dir, f"input.{lang}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        console.print(f"[green]Subtitle saved:[/green] {os.path.basename(srt_path)} ({len(snippets)} segments)")
        return srt_path

    except Exception as e:
        console.print(f"[yellow]YouTube transcript download failed:[/yellow] {e}")
        return None


def _transcript_entries_to_srt(entries) -> str:
    """Convert youtube-transcript-api entries to SRT format.

    Each entry: {'text': '...', 'start': 0.0, 'duration': 5.5}
    """
    lines = []
    for i, entry in enumerate(entries, 1):
        # Handle both dict and FetchedTranscriptSnippet objects
        if hasattr(entry, "text"):
            text = entry.text
            start = entry.start
            duration = entry.duration
        else:
            text = entry["text"]
            start = entry["start"]
            duration = entry["duration"]

        end = start + duration
        start_ts = _format_srt_time(start)
        end_ts = _format_srt_time(end)
        # Clean up text (YouTube sometimes has \n or HTML entities)
        text = text.replace("\n", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


# ── Bilibili subtitle download via bilibili-api ──

def _download_bilibili_subtitles(url: str, config: dict, temp_dir: str) -> str | None:
    """Download Bilibili subtitles using bilibili-api-python.

    Returns path to SRT file, or None if unavailable.
    """
    try:
        from bilibili_api import video, Credential
    except ImportError:
        console.print("[dim]bilibili-api-python not installed, trying yt-dlp for subtitles[/dim]")
        return None

    bvid = extract_bilibili_bvid(url)
    if not bvid:
        return None

    try:
        import asyncio

        # Build credential from cookies if available
        dl_config = config.get("download", {})
        credential = None
        cookies_file = dl_config.get("cookies_file", "")
        bilibili_config = dl_config.get("bilibili", {})

        sessdata = bilibili_config.get("sessdata", "")
        bili_jct = bilibili_config.get("bili_jct", "")
        buvid3 = bilibili_config.get("buvid3", "")

        if sessdata and bili_jct:
            credential = Credential(sessdata=sessdata, bili_jct=bili_jct, buvid3=buvid3)

        v = video.Video(bvid=bvid, credential=credential)

        async def _fetch():
            info = await v.get_info()
            subtitle_list = info.get("subtitle", {}).get("list", [])
            if not subtitle_list:
                return None

            # Pick first available subtitle
            sub_info = subtitle_list[0]
            lang = sub_info.get("lan", "unknown")
            sub_url = sub_info.get("subtitle_url", "")

            if not sub_url:
                return None

            if sub_url.startswith("//"):
                sub_url = "https:" + sub_url

            console.print(f"[green]Found Bilibili subtitle:[/green] {sub_info.get('lan_doc', lang)} ({lang})")

            # Download subtitle JSON
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(sub_url) as resp:
                    sub_data = await resp.json()

            return sub_data, lang

        result = asyncio.run(_fetch())
        if result is None:
            console.print("[yellow]No Bilibili subtitle available[/yellow]")
            return None

        sub_data, lang = result

        # Convert Bilibili JSON subtitle to SRT
        srt_content = _bilibili_json_to_srt(sub_data)
        srt_path = os.path.join(temp_dir, f"input.{lang}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        body = sub_data.get("body", [])
        console.print(f"[green]Subtitle saved:[/green] {os.path.basename(srt_path)} ({len(body)} segments)")
        return srt_path

    except Exception as e:
        console.print(f"[yellow]Bilibili subtitle download failed:[/yellow] {e}")
        return None


def _bilibili_json_to_srt(sub_data: dict) -> str:
    """Convert Bilibili subtitle JSON format to SRT.

    Bilibili format: {"body": [{"from": 0.0, "to": 3.5, "content": "..."}, ...]}
    """
    body = sub_data.get("body", [])
    lines = []
    for i, entry in enumerate(body, 1):
        start = entry.get("from", 0.0)
        end = entry.get("to", 0.0)
        text = entry.get("content", "").strip()
        if not text:
            continue
        lines.append(f"{i}")
        lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


# ── Shared utilities ──

def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _find_sidecar_subtitle(video_path: str) -> str | None:
    """Look for a subtitle file next to the video (e.g. video.srt, video.en.srt)."""
    base = Path(video_path).with_suffix("")
    for ext in (".srt", ".vtt", ".ass"):
        candidate = str(base) + ext
        if os.path.isfile(candidate):
            console.print(f"[green]Found sidecar subtitle:[/green] {candidate}")
            return candidate
        matches = glob.glob(f"{base}.*{ext}")
        if matches:
            console.print(f"[green]Found sidecar subtitle:[/green] {matches[0]}")
            return matches[0]
    return None


# ── Video download with subtitle fallback chain ──

def _download_url(url: str, config: dict, temp_dir: str) -> tuple[str, str | None]:
    """Download video + subtitles from URL.

    Subtitle strategy:
      1. Try dedicated API (youtube-transcript-api / bilibili-api)
      2. Fall back to yt-dlp embedded subtitle download
      3. None → pipeline will use Whisper
    """
    import yt_dlp

    # Step 1: Try dedicated subtitle APIs first (before video download)
    sub_path = None

    if is_youtube_url(url):
        console.print("[cyan]Trying youtube-transcript-api...[/cyan]")
        sub_path = _download_youtube_subtitles(url, temp_dir)
    elif is_bilibili_url(url):
        console.print("[cyan]Trying bilibili-api subtitle...[/cyan]")
        sub_path = _download_bilibili_subtitles(url, config, temp_dir)

    # Step 2: Download video (with yt-dlp subtitle fallback)
    dl_config = config.get("download", {})
    output_template = os.path.join(temp_dir, "input.%(ext)s")

    ydl_opts = {
        "format": dl_config.get("format", "bestvideo+bestaudio/best"),
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }

    # If we didn't get subtitles from dedicated API, let yt-dlp try
    if sub_path is None:
        ydl_opts.update({
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "srt/vtt/best",
            "subtitleslangs": ["en", "en-US", "en-orig", "zh-Hans", "zh-Hant", "ja", "ko"],
        })

    proxy = dl_config.get("proxy", "")
    if proxy:
        ydl_opts["proxy"] = proxy

    cookies_file = dl_config.get("cookies_file", "")
    if cookies_file and os.path.isfile(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    if is_bilibili_url(url):
        console.print("[cyan]Detected Bilibili URL[/cyan]")
        ydl_opts["format"] = "bestvideo+bestaudio/best"

    console.print(f"[yellow]Downloading video from:[/yellow] {url}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

        mp4_path = Path(filename).with_suffix(".mp4")
        if mp4_path.exists():
            video_path = str(mp4_path)
        else:
            video_path = None
            for f in Path(temp_dir).iterdir():
                if f.name.startswith("input") and f.suffix in (".mp4", ".mkv", ".webm"):
                    video_path = str(f)
                    break
            if not video_path:
                raise RuntimeError("Failed to download video.")

        # If we still don't have subtitles, check yt-dlp downloaded ones
        if sub_path is None:
            sub_path = _find_ytdlp_subtitle(temp_dir)

        if sub_path:
            console.print(f"[bold green]Subtitle ready:[/bold green] {os.path.basename(sub_path)}")
        else:
            console.print("[yellow]No subtitle found — will use Whisper transcription[/yellow]")

        return video_path, sub_path


def _find_ytdlp_subtitle(temp_dir: str) -> str | None:
    """Find subtitle files downloaded by yt-dlp in temp dir."""
    sub_files = []
    for f in Path(temp_dir).iterdir():
        if f.suffix in (".srt", ".vtt", ".ass"):
            sub_files.append(f)

    if not sub_files:
        return None

    # Prefer manual subs over auto-generated
    manual = [f for f in sub_files if "-auto" not in f.stem]
    auto = [f for f in sub_files if "-auto" in f.stem]

    best = manual[0] if manual else auto[0] if auto else None
    if best:
        console.print(f"[green]Found yt-dlp subtitle:[/green] {best.name}")
        return str(best)
    return None
