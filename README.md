# Video Translate

One-click translate any video into another language — with voice cloning, speaker gender matching, and subtitle-first optimization.

Translate videos from YouTube, Bilibili, or local files. The pipeline automatically downloads subtitles when available (skipping slow speech recognition), detects speaker gender, and generates natural-sounding translated audio.

一键翻译任何视频到另一种语言 — 支持声音克隆、说话人性别匹配、字幕优先优化。支持 YouTube、Bilibili 和本地视频文件。

## Status / 项目状态

The author has successfully tested Chinese ↔ English translation on both YouTube and Bilibili videos. Other language pairs are supported but have not been thoroughly tested — your mileage may vary.

作者已成功跑通 YouTube 和 Bilibili 的中英互译。其他语言对理论支持但未经充分测试，效果可能有所不同。

This project was developed with the assistance of LLM-based coding tools. Bugs are expected — PRs are welcome!

本项目的开发过程中使用了 LLM Coding 辅助。如发现 bug，欢迎提交 PR 修复！

---

## How It Works / 工作原理

```
Video Input ──→ Extract Audio ──→ Transcribe ──→ Translate ──→ TTS ──→ Compose ──→ Output Video
  (URL/file)      (ffmpeg)      (Whisper/subs)   (LLM/NMT)  (Edge/   (numpy)    (ffmpeg merge)
 视频输入        提取音频       语音识别/字幕     翻译文本    Clone)   合成音频     输出视频
```

**7-step pipeline / 7步流水线：**

| Step / 步骤 | What / 功能 | How / 实现 |
|------|------|-----|
| 1. Get Video / 获取视频 | Download or copy source video / 下载或拷贝源视频 | yt-dlp / local copy |
| 2. Extract Audio / 提取音频 | Get audio track from video / 从视频提取音轨 | ffmpeg |
| 3. Transcribe / 语音转文字 | Get timed text from speech / 获取带时间戳的文本 | Existing subtitles (preferred) or Whisper / 优先已有字幕，否则 Whisper |
| 4. Translate / 翻译 | Translate each segment / 逐段翻译 | Ollama / vLLM / HuggingFace / OpenAI / Claude |
| 5. TTS / 语音合成 | Generate translated speech / 生成翻译后的语音 | Edge TTS / CosyVoice2 / Fish Speech |
| 6. Compose / 合成音频 | Place audio segments at correct timestamps / 按时间戳放置音频片段 | numpy + soundfile |
| 7. Merge / 合并输出 | Replace audio track in video / 替换视频音轨 | ffmpeg |

**Smart features / 智能特性：**
- **Subtitle-first / 字幕优先**: Downloads existing subtitles from YouTube/Bilibili before falling back to Whisper — much faster and more accurate / 优先下载 YouTube/Bilibili 已有字幕，无需 Whisper，更快更准
- **Gender detection / 性别检测**: Uses wav2vec2 model to detect speaker gender, then selects matching male/female TTS voice / 用 wav2vec2 模型检测说话人性别，自动匹配同性别 TTS 语音
- **Caching / 缓存复用**: Reuses downloaded videos and extracted audio on reruns — only re-processes what changed / 重复运行时复用已下载的视频和音频，只重新处理变更部分

---

## Quick Start / 快速开始

### Option A: Local Install / 本地安装

```bash
# Clone the repo / 克隆仓库
git clone https://github.com/tianyu-z/translate-video.git
cd translate_video

# Option 1: Use the setup script / 使用安装脚本
bash start.sh
source venv/bin/activate    # Activate the venv created by start.sh

# Option 2: Install manually into your own environment / 手动安装
pip install -r requirements.txt

# Install PyTorch (required for gender detection and HuggingFace translation)
# 安装 PyTorch（性别检测和 HuggingFace 翻译后端需要）
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU only
# pip install torch   # Or with CUDA support / 或带CUDA支持
```

**Requirements / 系统要求:**
- Python 3.10+
- ffmpeg (`apt install ffmpeg` or `pip install static-ffmpeg` if no root access)
- PyTorch (for speaker gender detection and `huggingface` translation backend)
- ~2GB disk for models (Whisper small + translation model + gender detection)

### Option B: Docker

Works on CPU by default. For GPU acceleration, uncomment the `deploy` sections in `docker-compose.yaml` (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).

默认以 CPU 模式运行。如需 GPU 加速，取消 `docker-compose.yaml` 中 `deploy` 部分的注释（需安装 NVIDIA Container Toolkit）。

```bash
# Build / 构建
docker compose build

# The default translation backend is Ollama. Start it first:
# 默认翻译后端是 Ollama，需要先启动：
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:14b

# Translate a video / 翻译视频
docker compose run translate translate "https://youtube.com/watch?v=xxx" -t zh

# Or use HuggingFace backend (no Ollama needed) / 或使用 HuggingFace（无需Ollama）
docker compose run translate translate "https://youtube.com/watch?v=xxx" -t zh -b huggingface
```

> **Note / 注意:** When using Ollama in Docker, `config.yaml` must set the Ollama URL to the Docker service name, not `localhost`:
> Docker 环境下使用 Ollama 时，需要把配置里的地址改为 Docker 服务名：
> ```yaml
> translation:
>   ollama:
>     base_url: "http://ollama:11434"   # NOT localhost
> ```

---

## Usage / 使用方法

### Translate a video / 翻译视频

```bash
# YouTube video → Chinese / YouTube 视频翻译成中文
python main.py translate "https://www.youtube.com/watch?v=VIDEO_ID" -t zh

# Bilibili video → English / B站视频翻译成英文
python main.py translate "https://www.bilibili.com/video/BVxxxxxxx" -t en -s zh

# Local file → Japanese / 本地文件翻译成日文
python main.py translate video.mp4 -t ja

# With specific translation backend / 指定翻译后端
python main.py translate video.mp4 -t zh -b ollama
python main.py translate video.mp4 -t zh -b openai_api
python main.py translate video.mp4 -t zh -b huggingface

# Test with limited segments (faster) / 测试模式（只翻译前N段）
python main.py translate video.mp4 -t zh --max-segments 10
```

### All CLI options / 所有命令行选项

```
python main.py translate <source> [options]

Arguments:
  source                    Video file path or URL (YouTube/Bilibili)
                            视频文件路径或URL

Options:
  -t, --target-lang TEXT    Target language code (zh, en, ja, ko, fr, de, ...)
                            目标语言代码
  -s, --source-lang TEXT    Source language (default: auto-detect)
                            源语言（默认自动检测）
  -b, --backend TEXT        Translation backend
                            翻译后端 (ollama/vllm/sglang/huggingface/openai_api/claude_api)
  --tts TEXT                TTS backend (edge_tts/cosyvoice/fish_speech)
                            TTS 后端
  -m, --model TEXT          Override translation model name
                            覆盖翻译模型名称
  -o, --output TEXT         Output file path
                            输出文件路径
  -c, --config TEXT         Config file path (default: config.yaml)
                            配置文件路径
  --max-segments INT        Limit segments to process (0=all, for testing)
                            限制处理的段数（用于测试）
```

### Other commands / 其他命令

```bash
# List available TTS voices / 查看可用TTS语音
python main.py list-voices

# List translation backends / 查看翻译后端列表
python main.py list-backends
```

---

## Translation Backends / 翻译后端

| Backend | Description / 说明 | Requirements / 依赖 |
|---------|-----------|--------------|
| `ollama` | Local LLM via Ollama / 本地 Ollama 大模型 | Ollama server running |
| `vllm` | Local LLM via vLLM / 本地 vLLM 大模型 | vLLM server running |
| `sglang` | Local LLM via SGLang / 本地 SGLang 大模型 | SGLang server running |
| `huggingface` | Helsinki-NLP opus-mt models / 离线翻译模型 | Auto-downloads model |
| `openai_api` | OpenAI GPT-4o etc. | `OPENAI_API_KEY` |
| `claude_api` | Anthropic Claude | `ANTHROPIC_API_KEY` |

**HuggingFace backend** automatically selects the right model based on language pair (e.g., `Helsinki-NLP/opus-mt-zh-en` for Chinese→English). No API key needed — runs fully offline.

HuggingFace 后端根据语言对自动选择模型，无需API密钥，完全离线运行。

---

## TTS Backends / 语音合成后端

| Backend | Voice Cloning / 声音克隆 | Quality / 质量 | Speed / 速度 | Requirements / 依赖 |
|---------|:---:|:---:|:---:|------|
| `edge_tts` | No | Good | Fast | Free, no setup |
| `cosyvoice` | Yes | Excellent | Medium | CosyVoice2 server |
| `fish_speech` | Yes | Excellent | Medium | Fish Speech server |

**Edge TTS** is the default — free Microsoft TTS with 16 languages, automatic male/female voice selection based on speaker gender detection.

Edge TTS 是默认选项 — 免费的微软TTS，支持16种语言，根据说话人性别自动选择男/女声。

**Supported languages for Edge TTS / 支持的Edge TTS语言:**

| Language | Male Voice | Female Voice |
|----------|-----------|-------------|
| Chinese (zh) | YunxiNeural | XiaoxiaoNeural |
| English (en) | GuyNeural | AriaNeural |
| Japanese (ja) | KeitaNeural | NanamiNeural |
| Korean (ko) | InJoonNeural | SunHiNeural |
| French (fr) | HenriNeural | DeniseNeural |
| German (de) | ConradNeural | KatjaNeural |
| Spanish (es) | AlvaroNeural | ElviraNeural |
| Portuguese (pt) | AntonioNeural | FranciscaNeural |
| Russian (ru) | DmitryNeural | SvetlanaNeural |
| Arabic (ar) | HamedNeural | ZariyahNeural |
| Hindi (hi) | MadhurNeural | SwaraNeural |
| Italian (it) | DiegoNeural | ElsaNeural |
| Thai (th) | NiwatNeural | PremwadeeNeural |
| Vietnamese (vi) | NamMinhNeural | HoaiMyNeural |
| Indonesian (id) | ArdiNeural | GadisNeural |
| Turkish (tr) | AhmetNeural | EmelNeural |

---

## Configuration / 配置

All settings are in `config.yaml`. Key sections:

所有设置在 `config.yaml` 中，主要配置项：

```yaml
# Language / 语言
source_language: "auto"      # auto-detect or specify: en, zh, ja, ...
target_language: "zh"        # target language code

# Translation backend / 翻译后端
translation:
  backend: "ollama"          # ollama / vllm / sglang / huggingface / openai_api / claude_api
  ollama:
    base_url: "http://localhost:11434"  # Use "http://ollama:11434" in Docker
    model: "qwen2.5:14b"

# TTS / 语音合成
tts:
  backend: "edge_tts"        # edge_tts / cosyvoice / fish_speech

# Whisper (when no subtitles available) / Whisper语音识别
transcription:
  model: "small"             # tiny / base / small / medium / large-v3
  device: "auto"             # auto / cpu / cuda

# Output / 输出
output:
  keep_original_audio: false  # mix original audio in background
  original_audio_volume: 0.0  # 0.0=mute, 1.0=full
  generate_subtitles: true    # export .srt subtitle file

# Download / 下载
download:
  proxy: ""                   # HTTP proxy for YouTube/Bilibili
  cookies_file: ""            # cookies.txt for age-restricted videos
```

---

## Project Structure / 项目结构

```
translate_video/
├── main.py              # CLI entry point / 命令行入口
├── pipeline.py          # 7-step pipeline orchestration / 流水线编排
├── config.yaml          # Configuration / 配置文件
├── requirements.txt     # Python dependencies / Python依赖
├── start.sh             # Quick start script / 快速启动脚本
├── Dockerfile           # Docker image / Docker镜像
├── docker-compose.yaml  # Docker Compose (with Ollama) / Docker编排
└── modules/
    ├── downloader.py    # Video download + subtitle extraction / 视频下载+字幕提取
    ├── audio.py         # Audio extraction, speed adjustment, merging / 音频处理
    ├── transcriber.py   # Whisper STT + subtitle parsing (SRT/VTT/ASS) / 语音识别+字幕解析
    ├── translator.py    # Multi-backend translation / 多后端翻译
    ├── tts.py           # TTS + gender detection / 语音合成+性别检测
    └── composer.py      # Audio composition with timing / 音频合成
```

---

## Examples / 示例

### Translate a YouTube lecture to Chinese / 把YouTube讲座翻译成中文

```bash
python main.py translate "https://www.youtube.com/watch?v=VIDEO_ID" -t zh -b huggingface
```

Pipeline will: download video → grab YouTube subtitles (skip Whisper) → translate en→zh with opus-mt → detect speaker gender → generate Chinese TTS → output video.

### Translate a Bilibili video to English / 把B站视频翻译成英文

```bash
python main.py translate "https://www.bilibili.com/video/BVxxxxxxx" -t en -s zh -b huggingface
```

### Use Ollama for higher quality translation / 用Ollama获得更高质量翻译

```bash
# Start Ollama and pull a model first / 先启动Ollama并拉取模型
ollama pull qwen2.5:14b

python main.py translate video.mp4 -t zh -b ollama -m qwen2.5:14b
```

### Use OpenAI API / 使用OpenAI API翻译

```bash
export OPENAI_API_KEY="sk-..."
python main.py translate video.mp4 -t ja -b openai_api
```

### Docker workflow / Docker工作流

```bash
# Build / 构建
docker compose build

# First, update config.yaml for Docker networking:
# 先修改 config.yaml 适配 Docker 网络：
#   ollama.base_url: "http://ollama:11434"

# Start Ollama and pull a model / 启动Ollama并拉取模型
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:14b

# Put your video in ./input/, then translate / 放视频到 ./input/ 然后翻译
docker compose run translate translate /app/input/video.mp4 -t zh

# Or skip Ollama entirely with HuggingFace / 或用 HuggingFace 跳过Ollama
docker compose run translate translate /app/input/video.mp4 -t zh -b huggingface

# Output is in ./output/
```

---

## Subtitle Strategy / 字幕策略

The pipeline prioritizes existing subtitles over speech recognition for speed and accuracy:

流水线优先使用现有字幕，而非语音识别（更快更准确）：

| Priority / 优先级 | Source / 来源 | Method / 方法 |
|---------|--------|--------|
| 1 | YouTube subtitles | `youtube-transcript-api` |
| 1 | Bilibili subtitles | `bilibili-api-python` |
| 2 | yt-dlp embedded subs | yt-dlp `--write-sub` |
| 3 | Sidecar files | `video.srt` next to `video.mp4` |
| 4 | Whisper fallback | `faster-whisper` speech recognition |

---

## Troubleshooting / 常见问题

**ffmpeg not found / 找不到ffmpeg**
```bash
pip install static-ffmpeg   # Auto-downloads ffmpeg binary
```

**Whisper too slow on CPU / Whisper在CPU上太慢**
```yaml
# In config.yaml, use a smaller model:
transcription:
  model: "small"    # or "base", "tiny" for faster speed
```

**YouTube download blocked / YouTube下载被阻止**
```yaml
# In config.yaml, set a proxy and/or cookies:
download:
  proxy: "http://127.0.0.1:7890"
  cookies_file: "cookies.txt"   # Export from browser
```

**Bilibili subtitles need login / B站字幕需要登录**
```yaml
# In config.yaml, add Bilibili cookies:
download:
  bilibili:
    sessdata: "your_sessdata"
    bili_jct: "your_bili_jct"
    buvid3: "your_buvid3"
```

**Translation model not found / 翻译模型找不到**
- HuggingFace backend auto-selects `Helsinki-NLP/opus-mt-{src}-{tgt}`. Not all language pairs have models. For unsupported pairs, use `ollama` or `openai_api` instead.
- HuggingFace 后端自动选择翻译模型，但不是所有语言对都有模型。不支持的语言对请用 `ollama` 或 `openai_api`。

---

## License

MIT
