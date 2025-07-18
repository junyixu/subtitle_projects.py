#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2025 Junyi Xu <jyxu@mail.ustc.edu.cn>
#
# Distributed under terms of the MIT license.
"""
Professional Subtitle Translation System
参考 gpt-subtrans 和 VideoCaptioner 的架构设计

项目特点:
- 模块化设计，易于扩展
- 项目管理系统，支持暂停恢复
- 智能批处理和错误恢复
- 多LLM提供商支持
- 完整的日志和进度追踪

依赖安装:
pip install openai anthropic google-generativeai pysrt aiohttp rich typer pydantic
"""

import asyncio
import json
import re
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
import logging
import hashlib

import pysrt
import typer
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
import aiohttp
from pydantic import BaseModel, Field

# LLM 客户端导入
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

console = Console()
app = typer.Typer(help="🎬 Professional Subtitle Translation System")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('subtitle_translator.log'),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)


class TranslationStatus(str, Enum):
    """翻译状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class LLMProvider(str, Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    ZHIPU = "zhipu"
    MOONSHOT = "moonshot"
    QWEN = "qwen"
    OLLAMA = "ollama"


@dataclass
class SubtitleSegment:
    """字幕片段数据类"""
    index: int
    start: timedelta
    end: timedelta
    text: str
    translation: str = ""
    status: TranslationStatus = TranslationStatus.PENDING
    retry_count: int = 0
    error_message: str = ""

    @property
    def duration(self) -> float:
        return (self.end - self.start).total_seconds()

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def hash(self) -> str:
        """生成片段的唯一标识"""
        content = f"{self.index}_{self.start}_{self.end}_{self.text}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict:
        """转换为字典，用于JSON序列化"""
        return {
            'index': self.index,
            'start': self.start.total_seconds(),
            'end': self.end.total_seconds(),
            'text': self.text,
            'translation': self.translation,
            'status': self.status.value,
            'retry_count': self.retry_count,
            'error_message': self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SubtitleSegment':
        """从字典创建对象"""
        return cls(index=data['index'],
                   start=timedelta(seconds=data['start']),
                   end=timedelta(seconds=data['end']),
                   text=data['text'],
                   translation=data.get('translation', ''),
                   status=TranslationStatus(data.get('status', 'pending')),
                   retry_count=data.get('retry_count', 0),
                   error_message=data.get('error_message', ''))


class ProjectConfig(BaseModel):
    """项目配置模型"""
    name: str
    source_file: str
    target_language: str = "chinese"
    llm_provider: LLMProvider
    model: str
    temperature: float = 0.3
    max_tokens: int = 2000
    batch_size: int = 10
    max_retry: int = 3
    context_window: int = 15
    custom_prompt: Optional[str] = None
    glossary: Dict[str, str] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class TranslationProject:
    """翻译项目管理类"""

    def __init__(self, config: ProjectConfig, project_dir: Path):
        self.config = config
        self.project_dir = project_dir
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # 项目文件路径
        self.segments_file = self.project_dir / "segments.json"
        self.progress_file = self.project_dir / "progress.json"
        self.config_file = self.project_dir / "config.json"
        self.log_file = self.project_dir / "translation.log"

        # 内部状态
        self.segments: List[SubtitleSegment] = []
        self.start_time: Optional[datetime] = None
        self.pause_time: Optional[datetime] = None
        self.total_cost: float = 0.0

        # 设置项目级日志
        self.setup_project_logger()

    def setup_project_logger(self):
        """设置项目专用日志"""
        self.logger = logging.getLogger(f"project_{self.config.name}")
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def save_config(self):
        """保存项目配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config.model_dump(),
                      f,
                      indent=2,
                      ensure_ascii=False)

    def load_segments(self) -> bool:
        """加载字幕片段"""
        if self.segments_file.exists():
            with open(self.segments_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.segments = [
                    SubtitleSegment.from_dict(item) for item in data
                ]
            return True
        return False

    def save_segments(self):
        """保存字幕片段"""
        data = [seg.to_dict() for seg in self.segments]
        with open(self.segments_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_progress(self) -> Dict:
        """加载进度信息"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_progress(self):
        """保存进度信息"""
        progress_data = {
            'total_segments':
            len(self.segments),
            'completed_segments':
            len([
                s for s in self.segments
                if s.status == TranslationStatus.COMPLETED
            ]),
            'failed_segments':
            len([
                s for s in self.segments
                if s.status == TranslationStatus.FAILED
            ]),
            'start_time':
            self.start_time.isoformat() if self.start_time else None,
            'pause_time':
            self.pause_time.isoformat() if self.pause_time else None,
            'total_cost':
            self.total_cost,
            'completion_rate':
            self.get_completion_rate()
        }

        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)

    def get_completion_rate(self) -> float:
        """获取完成率"""
        if not self.segments:
            return 0.0
        completed = len([
            s for s in self.segments if s.status == TranslationStatus.COMPLETED
        ])
        return completed / len(self.segments) * 100

    def get_pending_segments(self) -> List[SubtitleSegment]:
        """获取待翻译的片段"""
        return [
            s for s in self.segments if s.status in
            [TranslationStatus.PENDING, TranslationStatus.FAILED]
        ]

    def get_failed_segments(self) -> List[SubtitleSegment]:
        """获取失败的片段"""
        return [
            s for s in self.segments if s.status == TranslationStatus.FAILED
        ]


class LLMClientManager:
    """LLM客户端管理器"""

    def __init__(self):
        self.providers = {
            LLMProvider.OPENAI: self._create_openai_client,
            LLMProvider.ANTHROPIC: self._create_anthropic_client,
            LLMProvider.GEMINI: self._create_gemini_client,
            LLMProvider.DEEPSEEK: self._create_openai_compatible_client,
            LLMProvider.ZHIPU: self._create_openai_compatible_client,
            LLMProvider.MOONSHOT: self._create_openai_compatible_client,
            LLMProvider.QWEN: self._create_openai_compatible_client,
            LLMProvider.OLLAMA: self._create_ollama_client,
        }

        self.endpoints = {
            LLMProvider.DEEPSEEK: "https://poloai.top/v1",
            LLMProvider.ZHIPU: "https://open.bigmodel.cn/api/paas/v4",
            LLMProvider.MOONSHOT: "https://api.moonshot.cn/v1",
            LLMProvider.QWEN:
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        }

        self.default_models = {
            LLMProvider.OPENAI: "gpt-4o-mini",
            LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            LLMProvider.GEMINI: "gemini-1.5-flash",
            LLMProvider.DEEPSEEK: "deepseek-v3",
            LLMProvider.ZHIPU: "glm-4-air",
            LLMProvider.MOONSHOT: "moonshot-v1-8k",
            LLMProvider.QWEN: "qwen-turbo",
            LLMProvider.OLLAMA: "qwen2.5:14b"
        }

    def create_client(self, provider: LLMProvider, **kwargs):
        """创建LLM客户端"""
        if provider not in self.providers:
            raise ValueError(f"不支持的提供商: {provider}")

        # 为OpenAI兼容客户端传递provider参数
        if provider in [
                LLMProvider.DEEPSEEK, LLMProvider.ZHIPU, LLMProvider.MOONSHOT,
                LLMProvider.QWEN
        ]:
            return self.providers[provider](provider, **kwargs)
        else:
            return self.providers[provider](**kwargs)

    def _create_openai_client(self, api_key: str, **kwargs):
        """创建OpenAI客户端"""
        if not OPENAI_AVAILABLE:
            raise ImportError("请安装: pip install openai")
        return openai.AsyncOpenAI(api_key=api_key)

    def _create_anthropic_client(self, api_key: str, **kwargs):
        """创建Anthropic客户端"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("请安装: pip install anthropic")
        return anthropic.AsyncAnthropic(api_key=api_key)

    def _create_gemini_client(self, api_key: str, model: str = None, **kwargs):
        """创建Gemini客户端"""
        if not GEMINI_AVAILABLE:
            raise ImportError("请安装: pip install google-generativeai")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model or self.default_models[LLMProvider.GEMINI])

    def _create_openai_compatible_client(self, provider: LLMProvider,
                                         api_key: str, **kwargs):
        """创建OpenAI兼容客户端"""
        if not OPENAI_AVAILABLE:
            raise ImportError("请安装: pip install openai")

        base_url = self.endpoints[provider]
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _create_ollama_client(self,
                              base_url: str = "http://localhost:11434",
                              **kwargs):
        """创建Ollama客户端"""
        return {"base_url": base_url, "type": "ollama"}

    def get_default_model(self, provider: LLMProvider) -> str:
        """获取默认模型"""
        return self.default_models.get(provider, "unknown")


class TranslationEngine:
    """翻译引擎"""

    def __init__(self, project: TranslationProject):
        self.project = project
        self.config = project.config
        self.logger = project.logger

        # 初始化LLM客户端
        self.client_manager = LLMClientManager()
        self.client = None
        self.session = None  # 用于HTTP请求

        # 翻译统计
        self.api_calls = 0
        self.total_tokens = 0
        self.errors = []

    async def initialize(self, api_key: str):
        """初始化翻译引擎"""
        try:
            # 创建LLM客户端
            if self.config.llm_provider == LLMProvider.OLLAMA:
                self.client = self.client_manager.create_client(
                    self.config.llm_provider,
                    base_url="http://localhost:11434")
                self.session = aiohttp.ClientSession()
            else:
                self.client = self.client_manager.create_client(
                    self.config.llm_provider,
                    api_key=api_key,
                    model=self.config.model)

            self.logger.info(f"已初始化 {self.config.llm_provider} 客户端")

        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise

    def create_translation_prompt(self,
                                  segments: List[SubtitleSegment],
                                  context: str = "") -> str:
        """创建翻译提示词"""
        # 基础提示词
        base_prompt = f"""你是专业的{self.config.target_language}字幕翻译专家。

翻译要求：
1. 保持原意和语调，使用自然的中文表达
2. 考虑字幕显示限制，简洁准确
3. 保持时间段内的语义连贯性
4. 适应中文观众的文化背景
5. 保持角色对话的一致性

"""

        # 添加术语表
        if self.config.glossary:
            base_prompt += "\n专业术语对照:\n"
            for en, zh in self.config.glossary.items():
                base_prompt += f"- {en} → {zh}\n"

        # 添加自定义提示
        if self.config.custom_prompt:
            base_prompt += f"\n特殊要求：{self.config.custom_prompt}\n"

        # 添加上下文
        if context:
            base_prompt += f"\n上下文信息：{context}\n"

        # 添加待翻译内容
        base_prompt += "\n请翻译以下字幕，按编号顺序返回：\n\n"

        for segment in segments:
            base_prompt += f"{segment.index}. {segment.text}\n"

        base_prompt += "\n翻译结果（每行一个翻译）："

        return base_prompt

    async def translate_batch(self,
                              segments: List[SubtitleSegment],
                              context: str = "") -> List[str]:
        """批量翻译"""
        prompt = self.create_translation_prompt(segments, context)

        try:
            if self.config.llm_provider == LLMProvider.OPENAI:
                return await self._translate_openai(prompt, len(segments))
            elif self.config.llm_provider == LLMProvider.ANTHROPIC:
                return await self._translate_anthropic(prompt, len(segments))
            elif self.config.llm_provider == LLMProvider.GEMINI:
                return await self._translate_gemini(prompt, len(segments))
            elif self.config.llm_provider in [
                    LLMProvider.DEEPSEEK, LLMProvider.ZHIPU,
                    LLMProvider.MOONSHOT, LLMProvider.QWEN
            ]:
                return await self._translate_openai_compatible(
                    prompt, len(segments))
            elif self.config.llm_provider == LLMProvider.OLLAMA:
                return await self._translate_ollama(prompt, len(segments))

        except Exception as e:
            self.logger.error(f"批量翻译失败: {e}")
            self.errors.append(str(e))
            return [""] * len(segments)

    async def _translate_openai(self, prompt: str,
                                segment_count: int) -> List[str]:
        """OpenAI翻译"""
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature)

        self.api_calls += 1
        self.total_tokens += response.usage.total_tokens

        content = response.choices[0].message.content
        return self._parse_translation_response(content, segment_count)

    async def _translate_anthropic(self, prompt: str,
                                   segment_count: int) -> List[str]:
        """Anthropic翻译"""
        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }])

        self.api_calls += 1
        # Anthropic暂时没有usage信息，估算
        self.total_tokens += len(prompt) + self.config.max_tokens

        content = response.content[0].text
        return self._parse_translation_response(content, segment_count)

    async def _translate_gemini(self, prompt: str,
                                segment_count: int) -> List[str]:
        """Gemini翻译"""
        import asyncio

        def _sync_generate():
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                ))
            return response.text

        content = await asyncio.get_event_loop().run_in_executor(
            None, _sync_generate)

        self.api_calls += 1
        self.total_tokens += len(prompt) + len(content)  # 估算

        return self._parse_translation_response(content, segment_count)

    async def _translate_openai_compatible(self, prompt: str,
                                           segment_count: int) -> List[str]:
        """OpenAI兼容API翻译"""
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature)

        self.api_calls += 1
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens += response.usage.total_tokens
        else:
            self.total_tokens += len(prompt) + len(
                response.choices[0].message.content)

        content = response.choices[0].message.content
        return self._parse_translation_response(content, segment_count)

    async def _translate_ollama(self, prompt: str,
                                segment_count: int) -> List[str]:
        """Ollama翻译"""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }

        async with self.session.post(f"{self.client['base_url']}/api/generate",
                                     json=payload) as response:
            if response.status == 200:
                data = await response.json()
                content = data.get('response', '')

                self.api_calls += 1
                self.total_tokens += len(prompt) + len(content)

                return self._parse_translation_response(content, segment_count)
            else:
                raise Exception(f"Ollama API错误: {response.status}")

    def _parse_translation_response(self, content: str,
                                    expected_count: int) -> List[str]:
        """解析翻译响应"""
        lines = content.strip().split('\n')
        translations = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 移除可能的编号前缀
            line = re.sub(r'^\d+\.\s*', '', line)
            line = re.sub(r'^-\s*', '', line)  # 移除列表符号
            translations.append(line)

        # 确保返回正确数量的翻译
        while len(translations) < expected_count:
            translations.append("")

        return translations[:expected_count]

    async def close(self):
        """清理资源"""
        if self.session:
            await self.session.close()


class SubtitleTranslationSystem:
    """字幕翻译系统主类"""

    def __init__(self, workspace_dir: str = "subtitle_projects"):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(exist_ok=True)

        # 全局配置文件
        self.global_config_file = self.workspace / "global_config.json"
        self.load_global_config()

        # 初始化LLM客户端管理器
        self.client_manager = LLMClientManager()

    def load_global_config(self):
        """加载全局配置"""
        default_config = {
            "api_keys": {
                "openai": "",
                "anthropic": "",
                "gemini": "",
                "deepseek": "",
                "zhipu": "",
                "moonshot": "",
                "qwen": ""
            },
            "default_settings": {
                "llm_provider": "zhipu",
                "temperature": 0.3,
                "max_tokens": 2000,
                "batch_size": 10,
                "max_retry": 3
            }
        }

        # 从文件加载配置
        if self.global_config_file.exists():
            with open(self.global_config_file, 'r', encoding='utf-8') as f:
                self.global_config = json.load(f)
        else:
            self.global_config = default_config

        # 从环境变量覆盖API密钥
        env_mappings = {
            "openai": ["OPENAI_API_KEY", "OPENAI_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_KEY"],
            "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY", "DEEPSEEK_KEY"],
            "zhipu": ["ZHIPU_API_KEY", "ZHIPU_KEY", "BIGMODEL_API_KEY"],
            "moonshot": ["MOONSHOT_API_KEY", "MOONSHOT_KEY"],
            "qwen": ["QWEN_API_KEY", "QWEN_KEY", "DASHSCOPE_API_KEY"]
        }

        for provider, env_vars in env_mappings.items():
            for env_var in env_vars:
                env_value = os.getenv(env_var)
                if env_value:
                    self.global_config["api_keys"][provider] = env_value
                    break

        # 如果文件不存在，保存合并后的配置
        if not self.global_config_file.exists():
            self.save_global_config()

    def save_global_config(self):
        """保存全局配置"""
        with open(self.global_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.global_config, f, indent=2, ensure_ascii=False)

    def list_projects(self) -> List[str]:
        """列出所有项目"""
        projects = []
        for project_dir in self.workspace.iterdir():
            if project_dir.is_dir() and (project_dir / "config.json").exists():
                projects.append(project_dir.name)
        return sorted(projects)

    def create_project(self, name: str, source_file: str,
                       config: Dict) -> TranslationProject:
        """创建新项目"""
        # 验证项目名称
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            raise ValueError("项目名称只能包含字母、数字、下划线和横线")

        project_dir = self.workspace / name
        if project_dir.exists():
            raise ValueError(f"项目 '{name}' 已存在")

        # 合并全局默认配置
        merged_config = {**self.global_config["default_settings"], **config}

        project_config = ProjectConfig(name=name,
                                       source_file=source_file,
                                       **merged_config)

        project = TranslationProject(project_config, project_dir)
        project.save_config()

        # 加载字幕文件
        self._load_subtitle_file(project, source_file)

        console.print(f"[green]✅ 项目 '{name}' 创建成功")
        return project

    def load_project(self, name: str) -> TranslationProject:
        """加载现有项目"""
        project_dir = self.workspace / name
        config_file = project_dir / "config.json"

        if not config_file.exists():
            raise ValueError(f"项目 '{name}' 不存在")

        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        project_config = ProjectConfig(**config_data)
        project = TranslationProject(project_config, project_dir)

        # 加载已有的片段和进度
        project.load_segments()

        return project

    def _load_subtitle_file(self, project: TranslationProject, file_path: str):
        """加载字幕文件到项目"""
        try:
            # 尝试不同编码
            srt_file = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                try:
                    srt_file = pysrt.open(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if srt_file is None:
                raise ValueError(f"无法解码字幕文件: {file_path}")

            segments = []
            for item in srt_file:
                text = self._clean_subtitle_text(item.text)
                if text.strip():  # 跳过空字幕
                    # 正确转换 SubRipTime 到 timedelta
                    start_td = timedelta(milliseconds=item.start.ordinal)
                    end_td = timedelta(milliseconds=item.end.ordinal)

                    segment = SubtitleSegment(index=item.index,
                                              start=start_td,
                                              end=end_td,
                                              text=text)
                    segments.append(segment)

            project.segments = segments
            project.save_segments()

            project.logger.info(f"已加载 {len(segments)} 个字幕片段")

        except Exception as e:
            project.logger.error(f"加载字幕文件失败: {e}")
            raise

    def _clean_subtitle_text(self, text: str) -> str:
        """清理字幕文本"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除换行符
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()

    async def translate_project(self, project: TranslationProject,
                                api_key: str):
        """翻译项目"""
        engine = TranslationEngine(project)

        try:
            await engine.initialize(api_key)

            project.start_time = datetime.now()
            pending_segments = project.get_pending_segments()

            if not pending_segments:
                console.print("[yellow]⚠️  没有待翻译的片段")
                return

            console.print(f"[blue]🚀 开始翻译 {len(pending_segments)} 个片段...")

            # 创建进度条
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn(
                        "[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console,
                    transient=True) as progress:

                task = progress.add_task(f"[green]翻译中...",
                                         total=len(pending_segments))

                # 分批处理
                batch_size = project.config.batch_size
                for i in range(0, len(pending_segments), batch_size):
                    batch = pending_segments[i:i + batch_size]

                    # 生成上下文
                    context = self._generate_context(batch[0],
                                                     project.segments)

                    # 翻译批次
                    try:
                        translations = await engine.translate_batch(
                            batch, context)

                        # 应用翻译结果
                        for segment, translation in zip(batch, translations):
                            if translation.strip():
                                segment.translation = translation
                                segment.status = TranslationStatus.COMPLETED
                            else:
                                segment.status = TranslationStatus.FAILED
                                segment.error_message = "翻译为空"

                        progress.update(task, advance=len(batch))

                        # 保存进度
                        project.save_segments()
                        project.save_progress()

                        # 避免API限制
                        await asyncio.sleep(0.5)

                    except Exception as e:
                        project.logger.error(f"批次翻译失败: {e}")
                        for segment in batch:
                            segment.status = TranslationStatus.FAILED
                            segment.error_message = str(e)
                            segment.retry_count += 1

            # 完成统计
            completed = len([
                s for s in project.segments
                if s.status == TranslationStatus.COMPLETED
            ])
            failed = len([
                s for s in project.segments
                if s.status == TranslationStatus.FAILED
            ])

            project.total_cost = self._calculate_cost(
                engine.total_tokens, project.config.llm_provider)
            project.save_progress()

            # 显示结果
            self._show_translation_summary(project, engine, completed, failed)

        finally:
            await engine.close()

    def _generate_context(self, current_segment: SubtitleSegment,
                          all_segments: List[SubtitleSegment]) -> str:
        """生成上下文信息"""
        context_segments = []
        current_index = current_segment.index

        # 获取前面的已翻译片段作为上下文
        for segment in all_segments:
            if (segment.index < current_index
                    and segment.status == TranslationStatus.COMPLETED
                    and len(context_segments) < 5):
                context_segments.append(segment)

        if context_segments:
            context_text = " ".join([
                f"{seg.text} -> {seg.translation}"
                for seg in context_segments[-3:]
            ])
            return f"前文对照: {context_text}"

        return ""

    def _calculate_cost(self, tokens: int, provider: LLMProvider) -> float:
        """计算翻译成本（USD）"""
        cost_per_million = {
            LLMProvider.OPENAI: 0.15,  # gpt-4o-mini
            LLMProvider.ANTHROPIC: 3.0,  # claude-3.5-sonnet
            LLMProvider.GEMINI: 0.0,  # 免费额度
            LLMProvider.DEEPSEEK: 0.14,  # 约$0.14/M tokens
            LLMProvider.ZHIPU: 0.014,  # 约$0.014/M tokens
            LLMProvider.MOONSHOT: 1.68,  # 约$1.68/M tokens
            LLMProvider.QWEN: 0.042,  # 约$0.042/M tokens
            LLMProvider.OLLAMA: 0.0  # 免费
        }

        rate = cost_per_million.get(provider, 0.0)
        return (tokens / 1_000_000) * rate

    def _show_translation_summary(self, project: TranslationProject,
                                  engine: TranslationEngine, completed: int,
                                  failed: int):
        """显示翻译总结"""
        table = Table(title=f"翻译完成 - {project.config.name}")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green")

        table.add_row("总片段数", str(len(project.segments)))
        table.add_row("翻译成功", str(completed))
        table.add_row("翻译失败", str(failed))
        table.add_row("完成率", f"{project.get_completion_rate():.1f}%")
        table.add_row("API调用次数", str(engine.api_calls))
        table.add_row("总Token数", f"{engine.total_tokens:,}")
        table.add_row("预估成本", f"${project.total_cost:.4f}")

        console.print(table)

        if failed > 0:
            console.print(f"[yellow]⚠️  {failed} 个片段翻译失败，可以重试")

    def export_bilingual_subtitles(self, project: TranslationProject,
                                   output_path: str):
        """导出双语字幕"""
        srt_file = pysrt.SubRipFile()

        for segment in project.segments:
            if segment.status == TranslationStatus.COMPLETED:
                bilingual_text = f"{segment.text}\n{segment.translation}"
            else:
                bilingual_text = segment.text

            # 正确转换 timedelta 到 SubRipTime
            start_ms = int(segment.start.total_seconds() * 1000)
            end_ms = int(segment.end.total_seconds() * 1000)

            item = pysrt.SubRipItem(
                index=segment.index,
                start=pysrt.SubRipTime.from_ordinal(start_ms),
                end=pysrt.SubRipTime.from_ordinal(end_ms),
                text=bilingual_text)
            srt_file.append(item)

        srt_file.save(output_path, encoding='utf-8')
        console.print(f"[green]✅ 双语字幕已导出: {output_path}")


# CLI 命令定义
@app.command("list")
def list_projects():
    """📋 列出所有翻译项目"""
    system = SubtitleTranslationSystem()
    projects = system.list_projects()

    if not projects:
        console.print("[yellow]📁 没有找到翻译项目")
        console.print("💡 使用 'create' 命令创建新项目")
        return

    table = Table(title="📁 翻译项目列表")
    table.add_column("项目名称", style="cyan")
    table.add_column("状态", style="green")
    table.add_column("完成率", style="yellow")
    table.add_column("最后修改", style="blue")

    for project_name in projects:
        try:
            project = system.load_project(project_name)
            completion_rate = project.get_completion_rate()

            # 判断状态
            if completion_rate == 100:
                status = "[green]✅ 已完成[/green]"
            elif completion_rate > 0:
                status = "[yellow]🔄 进行中[/yellow]"
            else:
                status = "[blue]📝 待开始[/blue]"

            # 获取最后修改时间
            config_file = project.project_dir / "config.json"
            mod_time = datetime.fromtimestamp(config_file.stat().st_mtime)
            mod_time_str = mod_time.strftime("%Y-%m-%d %H:%M")

            table.add_row(project_name, status, f"{completion_rate:.1f}%",
                          mod_time_str)

        except Exception as e:
            table.add_row(project_name, "[red]❌ 错误[/red]", "N/A", "N/A")

    console.print(table)


@app.command("create")
def create_project(
        name: str = typer.Argument(..., help="项目名称"),
        source_file: str = typer.Argument(..., help="源字幕文件路径"),
        provider: LLMProvider = typer.Option(LLMProvider.ZHIPU,
                                             "--provider",
                                             "-p",
                                             help="LLM提供商"),
        model: str = typer.Option(None, "--model", "-m", help="模型名称"),
        batch_size: int = typer.Option(10, "--batch", "-b", help="批处理大小"),
        temperature: float = typer.Option(0.3, "--temp", "-t", help="翻译温度"),
):
    """🆕 创建新的翻译项目"""

    # 验证源文件
    if not Path(source_file).exists():
        console.print(f"[red]❌ 源文件不存在: {source_file}")
        raise typer.Exit(1)

    system = SubtitleTranslationSystem()

    # 获取默认模型
    if not model:
        model = system.client_manager.get_default_model(provider)

    config = {
        "llm_provider": provider,
        "model": model,
        "batch_size": batch_size,
        "temperature": temperature,
    }

    try:
        project = system.create_project(name, source_file, config)

        # 显示项目信息
        provider_display = provider.value if hasattr(
            provider, 'value') else str(provider)
        panel = Panel(f"""
[bold green]项目创建成功！[/bold green]

📁 项目名称: {name}
📄 源文件: {source_file}
🤖 LLM提供商: {provider_display}
🎯 模型: {model}
📊 字幕片段: {len(project.segments)}

💡 下一步:
   1. 设置API密钥: subtitle-translator config set-key {provider_display} <your-key>
   2. 开始翻译: subtitle-translator translate {name}
            """,
                      title="🎉 项目创建完成",
                      border_style="green")
        console.print(panel)

    except Exception as e:
        console.print(f"[red]❌ 创建项目失败: {e}")
        raise typer.Exit(1)


@app.command("translate")
def translate_project(
        project_name: str = typer.Argument(..., help="项目名称"),
        api_key: str = typer.Option(None, "--api-key", help="API密钥（优先级最高）"),
        resume: bool = typer.Option(False, "--resume", "-r", help="从上次中断处继续"),
):
    """🚀 开始翻译项目"""

    system = SubtitleTranslationSystem()

    try:
        project = system.load_project(project_name)

        # 获取API密钥 - 修复枚举值访问问题
        if not api_key:
            # 安全地获取provider名称
            provider_name = project.config.llm_provider
            if hasattr(provider_name, 'value'):
                provider_name = provider_name.value

            api_key = system.global_config["api_keys"].get(provider_name)

            if not api_key:
                console.print(f"[red]❌ 未设置 {provider_name} 的API密钥")
                console.print(
                    f"💡 使用命令设置: python subtitle-translator.py config set-key {provider_name} <your-key>"
                )
                raise typer.Exit(1)

        # 显示翻译信息
        if resume:
            console.print(f"[blue]🔄 恢复翻译项目: {project_name}")
        else:
            console.print(f"[blue]🚀 开始翻译项目: {project_name}")

        # 异步运行翻译
        asyncio.run(system.translate_project(project, api_key))

    except Exception as e:
        console.print(f"[red]❌ 翻译失败: {e}")
        logger.error(f"翻译项目失败: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command("export")
def export_subtitles(project_name: str = typer.Argument(..., help="项目名称"),
                     output_file: str = typer.Option(None,
                                                     "--output",
                                                     "-o",
                                                     help="输出文件路径"),
                     format_type: str = typer.Option(
                         "bilingual",
                         "--format",
                         "-f",
                         help="导出格式: bilingual, chinese, original")):
    """📤 导出翻译后的字幕"""

    system = SubtitleTranslationSystem()

    try:
        project = system.load_project(project_name)

        if not output_file:
            output_file = f"{project_name}_{format_type}.srt"

        if format_type == "bilingual":
            system.export_bilingual_subtitles(project, output_file)
        else:
            console.print(f"[yellow]⚠️  格式 '{format_type}' 暂不支持，使用双语格式")
            system.export_bilingual_subtitles(project, output_file)

    except Exception as e:
        console.print(f"[red]❌ 导出失败: {e}")
        raise typer.Exit(1)


@app.command("status")
def show_project_status(project_name: str = typer.Argument(..., help="项目名称")):
    """📊 查看项目状态"""

    system = SubtitleTranslationSystem()

    try:
        project = system.load_project(project_name)
        progress_data = project.load_progress()

        # 统计信息
        total = len(project.segments)
        completed = len([
            s for s in project.segments
            if s.status == TranslationStatus.COMPLETED
        ])
        failed = len([
            s for s in project.segments if s.status == TranslationStatus.FAILED
        ])
        pending = total - completed - failed

        # 创建状态表格
        table = Table(title=f"📊 项目状态 - {project_name}")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green")

        table.add_row("项目名称", project.config.name)

        # 安全地获取 provider 名称
        provider_name = project.config.llm_provider
        if hasattr(provider_name, 'value'):
            provider_name = provider_name.value
        table.add_row("LLM提供商", provider_name)

        table.add_row("模型", project.config.model)
        table.add_row("总片段数", str(total))
        table.add_row("已完成", f"{completed} ({completed/total*100:.1f}%)")
        table.add_row("翻译失败", str(failed))
        table.add_row("待处理", str(pending))

        if progress_data:
            if progress_data.get('start_time'):
                table.add_row("开始时间", progress_data['start_time'][:19])
            if progress_data.get('total_cost'):
                table.add_row("已花费", f"${progress_data['total_cost']:.4f}")

        console.print(table)

        # 显示失败的片段
        if failed > 0:
            console.print(f"\n[yellow]⚠️  失败的片段:")
            failed_segments = project.get_failed_segments()
            for seg in failed_segments[:5]:  # 只显示前5个
                console.print(
                    f"  {seg.index}: {seg.text[:50]}... ({seg.error_message})")

            if len(failed_segments) > 5:
                console.print(f"  ... 还有 {len(failed_segments) - 5} 个失败片段")

    except Exception as e:
        console.print(f"[red]❌ 获取状态失败: {e}")
        raise typer.Exit(1)


# 配置管理子命令
config_app = typer.Typer(help="⚙️  配置管理")
app.add_typer(config_app, name="config")


@config_app.command("set-key")
def set_api_key(provider: str = typer.Argument(..., help="LLM提供商"),
                api_key: str = typer.Argument(..., help="API密钥")):
    """🔑 设置API密钥"""

    system = SubtitleTranslationSystem()

    if provider not in system.global_config["api_keys"]:
        console.print(f"[red]❌ 不支持的提供商: {provider}")
        console.print(
            f"💡 支持的提供商: {', '.join(system.global_config['api_keys'].keys())}")
        raise typer.Exit(1)

    system.global_config["api_keys"][provider] = api_key
    system.save_global_config()

    console.print(f"[green]✅ 已设置 {provider} 的API密钥")


@config_app.command("show")
def show_config():
    """📋 显示当前配置"""

    system = SubtitleTranslationSystem()

    table = Table(title="⚙️  全局配置")
    table.add_column("提供商", style="cyan")
    table.add_column("API密钥状态", style="green")

    for provider, key in system.global_config["api_keys"].items():
        status = "[green]✅ 已设置[/green]" if key else "[red]❌ 未设置[/red]"
        table.add_row(provider, status)

    console.print(table)


if __name__ == "__main__":
    app()
