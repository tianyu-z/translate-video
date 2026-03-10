#!/usr/bin/env python3
"""Video Translation CLI - One-click translate videos to any language.

Usage:
    python main.py translate <source> [options]
    python main.py translate video.mp4 --target zh
    python main.py translate "https://youtube.com/watch?v=xxx" --target ja
    python main.py translate "https://bilibili.com/video/BVxxx" --target en

    python main.py list-voices          # List available Edge TTS voices
    python main.py list-backends        # List available translation backends
"""

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli():
    """Video Translation Tool - Translate videos to any language."""
    pass


@cli.command()
@click.argument("source")
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--source-lang", "-s", default=None, help="Source language (default: auto-detect)")
@click.option("--target-lang", "-t", default=None, help="Target language code (e.g., zh, en, ja)")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--backend", "-b", default=None,
              help="Translation backend (ollama/vllm/sglang/huggingface/openai_api/claude_api/codex)")
@click.option("--tts", default=None, help="TTS backend (edge_tts/cosyvoice/fish_speech)")
@click.option("--model", "-m", default=None, help="Override translation model name")
@click.option("--max-segments", default=0, type=int, help="Limit segments to translate (0=all, for testing)")
@click.option("--codex-relogin", is_flag=True, default=False,
              help="Force Codex re-login (delete cached token and re-authenticate)")
def translate(source, config, source_lang, target_lang, output, backend, tts, model, max_segments, codex_relogin):
    """Translate a video file or URL."""
    import yaml

    # Handle Codex re-login if requested
    if codex_relogin:
        import subprocess
        from pathlib import Path

        auth_file = Path.home() / ".codex" / "auth.json"
        if auth_file.exists():
            auth_file.unlink()
            console.print("[yellow]Deleted cached Codex token[/yellow]")

        console.print("[cyan]Launching Codex login...[/cyan]")
        result = subprocess.run(["codex", "login"], check=False)
        if result.returncode != 0:
            console.print("[red]Codex login failed. Continuing anyway...[/red]")
        else:
            console.print("[green]Codex login successful![/green]")

    # Load and optionally override config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    if backend:
        cfg["translation"]["backend"] = backend
    if tts:
        cfg["tts"]["backend"] = tts
    if model:
        backend_name = cfg["translation"]["backend"]
        if backend_name in cfg["translation"]:
            cfg["translation"][backend_name]["model"] = model

    # Write temporary config if overrides were applied
    if any([backend, tts, model]):
        import tempfile
        tmp_config = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="vtrans_"
        )
        yaml.dump(cfg, tmp_config)
        tmp_config.close()
        config = tmp_config.name

    from pipeline import run_pipeline
    run_pipeline(
        source=source,
        config_path=config,
        source_lang=source_lang,
        target_lang=target_lang,
        output_path=output,
        max_segments=max_segments,
    )


@cli.command("list-voices")
def list_voices():
    """List available Edge TTS voices by language."""
    from modules.tts import EDGE_VOICES

    table = Table(title="Default Edge TTS Voices")
    table.add_column("Language", style="cyan")
    table.add_column("Male Voice", style="green")
    table.add_column("Female Voice", style="magenta")
    for lang, voices in sorted(EDGE_VOICES.items()):
        table.add_row(lang, voices["male"], voices["female"])
    console.print(table)
    console.print("\n[dim]For all voices, run: edge-tts --list-voices[/dim]")


@cli.command("list-backends")
def list_backends():
    """List available translation backends."""
    table = Table(title="Translation Backends")
    table.add_column("Backend", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Requires", style="yellow")

    rows = [
        ("ollama", "Local LLM via Ollama", "Ollama server running"),
        ("vllm", "Local LLM via vLLM (OpenAI-compatible)", "vLLM server running"),
        ("sglang", "Local LLM via SGLang (OpenAI-compatible)", "SGLang server running"),
        ("huggingface", "HuggingFace Transformers (local)", "Model downloaded"),
        ("openai_api", "OpenAI API (GPT-4o etc.)", "OPENAI_API_KEY"),
        ("claude_api", "Anthropic Claude API", "ANTHROPIC_API_KEY"),
        ("codex", "ChatGPT OAuth via Codex CLI (GPT-5.4)", "codex login (no API key needed)"),
    ]
    for name, desc, req in rows:
        table.add_row(name, desc, req)
    console.print(table)


if __name__ == "__main__":
    cli()
