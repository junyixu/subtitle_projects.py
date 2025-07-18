# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Professional Subtitle Translation System** - A modular Python CLI tool for translating subtitles using multiple LLM providers (OpenAI, Anthropic, Gemini, Zhipu, Qwen, etc.) with academic focus and physics terminology support.

## Architecture

### Core Components

- **subtitle-translator.py**: Main CLI application (~1200 lines)
- **TranslationProject**: Manages project state, segments, progress, and persistence
- **TranslationEngine**: Handles LLM interactions and batch processing
- **LLMClientManager**: Unified interface for multiple LLM providers
- **SubtitleSegment**: Data class for individual subtitle entries

### Key Design Patterns

- **Project-based workflow**: Each subtitle file becomes a managed project
- **State persistence**: JSON files store segments, progress, and config
- **Batch processing**: Configurable batch sizes for API efficiency
- **Error recovery**: Automatic retries with exponential backoff
- **Cost tracking**: Real-time cost estimation per provider

## Directory Structure

```
subtitle_projects/                 # Workspace directory
├── global_config.json            # API keys and defaults
├── templates/                    # Translation templates
│   ├── academic_lecture.json     # Physics-focused academic content
│   ├── course_series.json        # Consistency-first batch processing
│   └── quick_preview.json        # Free tier testing
├── glossaries/                   # Terminology databases
│   └── physics_terms.json        # Physics terminology mappings
└── <project_name>/              # Individual project directories
    ├── config.json               # Project-specific settings
    ├── segments.json             # Subtitle segments with translation state
    ├── progress.json             # Translation progress metrics
    └── translation.log           # Project-specific logs
```

## Common Commands

### Project Management
```bash
# Create new translation project
python subtitle-translator.py create <name> <source.srt> --provider zhipu

# List all projects
python subtitle-translator.py list

# Check project status
python subtitle-translator.py status <project_name>

# Start/resume translation
python subtitle-translator.py translate <project_name>

# Export bilingual subtitles
python subtitle-translator.py export <project_name> -o output.srt
```

### Configuration
```bash
# Set API keys
python subtitle-translator.py config set-key zhipu <your_key>
python subtitle-translator.py config set-key gemini <your_key>

# View current configuration
python subtitle-translator.py config show
```

### Academic Workflow
```bash
# Quick preview with free provider
python subtitle-translator.py create preview lecture.srt --provider gemini
python subtitle-translator.py translate preview

# Production translation with cost-effective provider
python subtitle-translator.py create final lecture.srt --provider zhipu
python subtitle-translator.py translate final
```

## Development Setup

### Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `pysrt`: SRT file parsing
- `typer`: CLI interface
- `rich`: Terminal output formatting
- `aiohttp`: Async HTTP client
- `pydantic`: Data validation
- LLM SDKs: `openai`, `anthropic`, `google-generativeai`

## LLM Provider Support

| Provider | Model | Cost (USD/M tokens) | Use Case |
|----------|--------|---------------------|----------|
| Zhipu | glm-4-air | $0.014 | Production (cheapest) |
| Qwen | qwen-turbo | $0.042 | Balanced quality/cost |
| Gemini | gemini-1.5-flash | $0.00 | Free testing |
| OpenAI | gpt-4o-mini | $0.15 | Standard quality |

## Project State Files

- **segments.json**: Array of SubtitleSegment objects with translation status
- **progress.json**: Completion metrics, timestamps, cost tracking
- **config.json**: Project-specific LLM settings and metadata

## Error Handling

- **Encoding detection**: Auto-detects subtitle file encoding (utf-8, gbk, etc.)
- **API failures**: Automatic retry with exponential backoff
- **Network issues**: Progress saved, can resume from interruption
- **Translation failures**: Individual segment retry without full restart

## Extension Points

- **New templates**: Add JSON files to `templates/` directory
- **Custom glossaries**: Add terminology JSON files to `glossaries/`
- **New LLM providers**: Extend `LLMClientManager` class
- **Custom prompts**: Modify template configurations