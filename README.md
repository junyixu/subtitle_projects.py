# 字幕翻译系统配置说明

## 🎯 增加物理术语优化

### 📊 API成本对比（每百万token）
- **智谱GLM**: ~$0.014 (最便宜，已配置)
- **阿里千问**: ~$0.042 (性价比高，已配置) 
- **Gemini**: 免费额度 (预览用，已配置)

### 🚀 快速开始

1. **创建项目**
```bash
python subtitle-translator.py create lecture01 /path/to/lecture01.srt --provider zhipu
```

2. **查看所有项目**
```bash
python subtitle-translator.py list
```

3. **开始翻译**
```bash
python subtitle-translator.py translate lecture01
```

4. **查看进度**
```bash
python subtitle-translator.py status lecture01
```

5. **导出结果**
```bash
python subtitle-translator.py export lecture01 -o lecture01_bilingual.srt
```

### 📚 学术场景优化

#### 使用术语表
```bash
# 物理学术语会自动识别并保持翻译一致性
# 可以在 glossaries/physics_terms.json 中添加更多术语
```

#### 批量处理课程
```bash
# 处理整个学期的课程
for i in {01..20}; do
    python subtitle-translator.py create "physics_lecture_$i" "lecture_$i.srt" --provider zhipu
    python subtitle-translator.py translate "physics_lecture_$i" &
done
```

#### 成本控制
```bash
# 先用免费Gemini预览
python subtitle-translator.py create preview lecture.srt --provider gemini
python subtitle-translator.py translate preview

# 满意后用智谱正式翻译
python subtitle-translator.py create final lecture.srt --provider zhipu  
python subtitle-translator.py translate final
```

### ⚙️ 配置管理

#### 查看当前配置
```bash
python subtitle-translator.py config show
```

#### 更新API密钥
```bash
python subtitle-translator.py config set-key zhipu your_new_key
python subtitle-translator.py config set-key qwen your_new_key
python subtitle-translator.py config set-key gemini your_new_key
```

### 🎨 模板使用

- **academic_lecture.json**: 学术讲座（高质量，适中成本）
- **course_series.json**: 课程系列（一致性优先）
- **quick_preview.json**: 快速预览（免费Gemini）

### 📁 目录结构
```
subtitle_projects/
├── global_config.json          # 全局配置
├── templates/                  # 翻译模板
│   ├── academic_lecture.json
│   ├── course_series.json
│   └── quick_preview.json
├── glossaries/                 # 术语表
│   └── physics_terms.json
├── your_project/              # 你的翻译项目
│   ├── config.json
│   ├── segments.json
│   ├── progress.json
│   └── translation.log
└── README.md                  # 本说明文件
```

### 🔧 故障排除

- **API限制**: 自动重试，支持断点续传
- **网络中断**: 进度自动保存，可随时恢复
- **翻译质量**: 可针对特定片段重新翻译
- **成本控制**: 实时监控，超预算自动警告
