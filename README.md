# Reunion Skill - 重逢 🕯️

> "死亡不是终点，遗忘才是。"

用 AI 的方式，让逝去的亲人以另一种形式继续陪伴。

---

## 项目简介

**Reunion Skill** 是一个基于 Claude Code 的本地化 AI 技能，通过对逝者生前非结构化数据（聊天记录、日记、照片等）的分析与提取，构建一个具备情感还原度的数字陪伴 Agent。

### 核心特点

- 🔒 **纯本地运行**：所有数据在本地处理，绝不上传云端
- 💬 **双层还原**：Memory（共同记忆）+ Persona（人物性格）
- 🎯 **个性化对话**：还原 ta 的说话方式、口头禅、语气词
- 🧠 **渐进式回忆**：根据对话上下文动态唤起相关记忆，避免一次性暴露所有内容
- 🔍 **智能检索**：RAG + 上下文管理，语义理解更精准
- 📈 **持续进化**：支持追加记忆、对话纠正
- 🛡️ **心理护栏**：内置情绪检测和安全干预机制

---

## 快速开始

### 1. 安装

将项目克隆到 Claude Code skills 目录：

```bash
# Windows
git clone https://github.com/yangdongchen66-boop/reunion-skill.git %USERPROFILE%\.claude\skills\reunion

# macOS/Linux
git clone https://github.com/yangdongchen66-boop/reunion-skill.git ~/.claude/skills/reunion
```

### 2. 使用

在 Claude Code 中输入：

```
/reunion-create
```

然后按照提示：
1. 输入称呼（如：奶奶、爷爷、猫猫）
2. 提供基本信息和性格印象
3. 导入原材料（聊天记录、日记、照片或直接口述）
4. 确认生成

### 3. 与 ta 对话

创建完成后，使用以下命令：

| 命令 | 功能 |
|------|------|
| `/{slug}` | 像 ta 一样跟你聊天 |
| `/{slug}-memory` | 查看共同记忆 |
| `/{slug}-persona` | 查看人设配置 |

---

## 主流程

```
Step 1: 基础信息录入（3 个问题）
    ↓
Step 2: 原材料导入（聊天记录/日记/照片/口述）
    ↓
Step 3: 双线分析（Memory + Persona）
    ↓
Step 4: 生成并预览
    ↓
Step 5: 写入文件并安装 Skill
```

---

## 项目结构

```
reunion-skill/
├── SKILL.md                 # 主流程定义
├── README.md                # 项目说明
├── requirements.txt         # Python 依赖
│
├── prompts/                 # Prompt 模板
│   ├── intake.md            # 信息采集
│   ├── memory_analyzer.md   # 记忆分析
│   ├── persona_analyzer.md  # 人设分析
│   ├── memory_builder.md    # 记忆构建
│   ├── persona_builder.md   # 人设构建
│   ├── merger.md            # 增量合并
│   ├── correction_handler.md # 对话纠正
│   └── farewell.md          # 告别仪式
│
├── core/                    # 核心模块
│   ├── data_parser.py       # 数据解析器
│   ├── persona_distiller.py # 人设蒸馏
│   ├── memory_store.py      # 记忆存储（RAG）
│   ├── chat_engine.py       # 对话引擎
│   └── safety_guard.py      # 安全护栏
│
├── tools/                   # 工具脚本
│   ├── wechat_parser.py     # 微信记录解析
│   └── version_manager.py   # 版本管理
│
└── reunions/                # 纪念对象数据（用户创建后生成）
    └── {slug}/
        ├── memory.md
        ├── persona.md
        ├── meta.json
        └── SKILL.md
```

---

## 数据格式

### 支持的原材料

| 类型 | 格式 | 说明 |
|------|------|------|
| 聊天记录 | .txt | 微信/QQ 导出的文本格式 |
| 日记/信件 | .txt/.md | 任意文本格式 |
| 照片 | .jpg/.png | 提取时间地点作为记忆锚点 |
| 口述 | 直接输入 | 交互式引导 |

### 生成的数据

- `memory.md`: 共同记忆（时间线、关键事件、日常点滴）
- `persona.md`: 人设配置（语言风格、价值观、情感表达）
- `meta.json`: 元信息（创建时间、版本、来源等）

---

## 进化模式

### 渐进式回忆（核心特性）

不同于传统 RAG 一次性返回所有相关记忆，渐进式回忆模拟真实人类：
- **话题触发**：用户提到某个话题时，检索相关记忆
- **自然涌现**：使用“突然想起...”、“记得吗？”等方式引入记忆
- **情感共鸣**：用户情绪低落时优先唤起温暖记忆
- **避免重复**：同一件事不会在短时间内多次提及

### 上下文感知检索

结合当前查询 + 对话历史进行智能检索：
- 维护最近 10 轮对话作为上下文
- 语义相似度 + 上下文权重融合排序
- 自动过滤已频繁提及的记忆

### 追加记忆

提供新的回忆材料后，自动分析并合并到现有数据中。

### 对话纠正

在对话中说 "ta 不会这样说" 或 "ta 应该是..."，系统会自动记录纠正并更新人设。

---

## 安全与隐私

- **纯本地运行**：所有数据处理在本地完成
- **不上传云端**：绝不发送任何数据到外部服务器
- **用户完全控制**：可随时删除或封存数据
- **心理护栏**：检测到高风险情绪时触发干预，提供心理援助热线

---

## 技术栈

- **Claude Code**: 主要运行环境
- **Python 3.8+**: 工具脚本
- **ChromaDB**: 向量数据库（可选，用于 RAG）
- **Sentence Transformers / ONNX Runtime**: 文本嵌入（可选，支持多种后端）
- **渐进式回忆系统**: 自研上下文感知记忆引擎

---

## 致谢

- Inspired by [colleague-skill](https://github.com/colleague-skill) - 感谢提供的技能开发思路和架构参考

---

## 许可证

MIT License

---

## 温馨提示

> 如果聊天过程中感到情绪波动，请适时休息。
> 
> **请珍惜眼前人。** ❤️

---

*他们从未离开，只是换了一种方式存在。*
