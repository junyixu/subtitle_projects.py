# GitHub Actions Workflows

This directory contains automated workflows for the subtitle translation system.

## Workflows Overview

### 1. subtitle-translation.yml
**Automated subtitle translation** triggered by:
- New `.srt` files in `subtitles/` directory
- Manual workflow dispatch
- Pull requests with subtitle changes

**Features:**
- Multi-provider support (Gemini, Zhipu, Qwen, OpenAI)
- Batch processing with configurable parameters
- Artifact uploads for translated files
- Cost-effective Gemini as default for testing

### 2. cost-monitoring.yml
**Budget control and cost tracking**:
- Daily cost analysis
- Budget limit enforcement (\$10/day default)
- GitHub issues for budget alerts
- Pull request comments for cost warnings

### 3. quality-check.yml
**Quality assurance pipeline**:
- Subtitle file validation (format, timing, encoding)
- Code quality checks (Black, isort, flake8)
- Security scanning (bandit, safety)
- Translation quality metrics

### 4. release.yml
**Release and deployment**:
- Multi-platform testing (Ubuntu, Windows, macOS)
- Python version compatibility (3.9-3.12)
- PyPI publishing on tagged releases
- Docker image building and pushing
- Automated changelog generation

## Setup Instructions

### Required Secrets

Configure these secrets in your GitHub repository:

1. **API Keys** (for translation providers):
   - `GEMINI_API_KEY` - Google Gemini API key
   - `ZHIPU_API_KEY` - Zhipu AI API key
   - `QWEN_API_KEY` - Alibaba Qwen API key
   - `OPENAI_API_KEY` - OpenAI API key

2. **Publishing** (optional):
   - `PYPI_API_TOKEN` - PyPI token for package publishing
   - `DOCKERHUB_USERNAME` - Docker Hub username
   - `DOCKERHUB_TOKEN` - Docker Hub access token

### Directory Structure

```
subtitles/          # Place new .srt files here for auto-translation
translations/       # Output directory for translated files
.github/workflows/
├── subtitle-translation.yml    # Main translation workflow
├── cost-monitoring.yml         # Budget control
├── quality-check.yml          # Quality assurance
└── release.yml                # Release automation
```

### Usage Examples

#### Automatic Translation
1. Push `.srt` files to `subtitles/` directory
2. Workflow automatically translates using Gemini (free tier)
3. Download translated files from workflow artifacts

#### Manual Translation
```bash
# Run via GitHub CLI
gh workflow run subtitle-translation.yml --ref main \
  -f subtitle_file="lecture01.srt" \
  -f provider="zhipu" \
  -f target_language="chinese"
```

#### Budget Monitoring
```bash
# Check current costs
gh workflow run cost-monitoring.yml --ref main -f check_budget=true
```

### Customization

#### Budget Limits
Edit `cost-monitoring.yml`:
```yaml
env:
  BUDGET_LIMIT: 20.00  # USD daily limit
  WARNING_THRESHOLD: 0.9  # 90% warning
```

#### Translation Providers
Modify provider options in `subtitle-translation.yml` workflow_dispatch inputs.

#### Quality Thresholds
Adjust validation rules in `quality-check.yml` for subtitle timing and format checks.

## Monitoring

### GitHub Issues
- Budget exceeded alerts
- Translation failures
- Quality warnings

### Pull Request Comments
- Automatic cost estimates
- Translation quality reports
- Security scan results

### Artifacts
- Translated subtitle files
- Cost reports
- Quality metrics
- Security scan results

## Troubleshooting

### Common Issues

1. **Translation not starting**: Check subtitle files are in `subtitles/` directory
2. **API errors**: Verify API keys are set in repository secrets
3. **Budget alerts**: Review cost monitoring workflow configuration
4. **Quality failures**: Check validation rules in quality-check.yml

### Debug Mode
Enable debug logging by setting `ACTIONS_STEP_DEBUG` secret to `true`.
