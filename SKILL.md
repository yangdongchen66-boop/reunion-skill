---
name: reunion
description: 重逢 - 让逝去的亲人以另一种方式继续陪伴
user-invocable: true
---

<!-- 
⚠️ 执行规则（减少用户确认次数）：
1. 文件读取操作：直接执行，无需确认
2. 目录创建和文件写入：批量执行，一次确认
3. 只在以下节点询问用户：
   - Step 1 结束后（确认基础信息）
   - Step 2 选择导入方式时
   - Step 4 预览后（确认生成）
4. 分析过程（Step 3）无需用户交互
-->

# Reunion Skill - 重逢

> "死亡不是终点，遗忘才是。"
> 
> 用 AI 的方式，让逝去的亲人以另一种形式继续陪伴。

---

## 主流程：创建纪念对象

### Step 1：基础信息录入（3 个问题）

参考 `${SKILL_DIR}/prompts/intake.md` 的问题序列，只问 3 个问题：

1. **称呼**（必填）
   - 你想怎么称呼 ta？
   - 示例：奶奶 / 爷爷 / 爸爸 / 妈妈 / 外婆

2. **基本信息**（一句话：年龄、职业、地域、离开多久了）
   - 示例：85岁 退休教师 山东人 走了三年了
   - 示例：78岁 农民 四川农村的 去年冬天走的

3. **性格印象**（一句话：ta是什么样的人、口头禅、生活习惯）
   - 示例：特别节俭 总说"吃饱饭比什么都重要" 爱唠叨但很疼我
   - 示例：脾气倔 但心软 每次打电话都问吃了没

除称呼外均可跳过。收集完后汇总确认再进入下一步。

---

### Step 2：原材料导入

询问用户：

> 原材料怎么提供？回忆越多，还原度越高。
> 
> [A] 聊天记录（txt 文件路径）
> [B] 日记/文字记录（文件路径）
> [C] 照片（图片路径）
> [D] 直接口述
> 
> 可以混用，也可以跳过。

**用户提供文件路径后，直接读取内容，无需额外确认。**

如选 D（口述），简单引导：
- ta 最常说的话？
- ta 的口头禅或习惯用语？
- 印象最深的事？

---

### Step 3：分析原材料

将收集到的所有原材料汇总，按两条线分析：

**线路 A（共同记忆 Memory）**：
- 参考 `${SKILL_DIR}/prompts/memory_analyzer.md`
- 提取：共同经历、日常习惯、饮食偏好、重要事件、温馨瞬间
- 建立时间线：出生 → 关键人生节点 → 你们的共同记忆 → 离开

**线路 B（人物性格 Persona）**：
- 参考 `${SKILL_DIR}/prompts/persona_analyzer.md`
- 提取：说话风格、价值观、情感表达方式、生活态度
- 生成 4 层人设结构：身份 → 语言风格 → 核心价值观 → 行为禁区

---

### Step 4：生成并预览

参考 `${SKILL_DIR}/prompts/memory_builder.md` 生成记忆内容。
参考 `${SKILL_DIR}/prompts/persona_builder.md` 生成人设内容。

向用户展示摘要：

```
共同记忆摘要：
  - 关系：{relationship}
  - 关键记忆：{xxx}
  - 常做的事：{xxx}
  - 常说的话：{xxx}
  ...

人设摘要：
  - 说话风格：{xxx}
  - 口头禅：{xxx}
  - 核心价值观：{xxx}
  - 表达关心的方式：{xxx}
  ...

确认生成？还是需要调整？
```

---

### Step 5：写入文件并安装

用户确认后，**一次性批量创建所有文件**（减少确认次数）：

#### 5.1 创建数据文件

目标目录：`reunions/{slug}/`

需要创建的文件（按顺序依次写入，不要分开确认）：
1. `memory.md` - 共同记忆
2. `persona.md` - 人设配置  
3. `meta.json` - 元信息
4. `SKILL.md` - 完整 Skill

#### 5.2 安装到 Claude Code skills 目录

**重要**：Claude Code 只通过 SKILL.md 的 `name` 字段匹配触发词。
为了让 `/{slug}`、`/{slug}-memory`、`/{slug}-persona` 都能工作，
需要创建 **3 个独立的 skill 目录**：

| 目录 | name 字段 | 触发词 | 功能 |
|------|------------|--------|------|
| `~/.claude/skills/{slug}/` | `{slug}` | `/{slug}` | 对话 |
| `~/.claude/skills/{slug}-memory/` | `{slug}-memory` | `/{slug}-memory` | 查看记忆 |
| `~/.claude/skills/{slug}-persona/` | `{slug}-persona` | `/{slug}-persona` | 查看人设 |

**执行步骤**（一次性完成，不要分开确认）：

1. 创建 `~/.claude/skills/{slug}/SKILL.md`：
   - frontmatter 中 `name: {slug}`
   - 包含完整的 PART A、PART B、运行规则

2. 创建 `~/.claude/skills/{slug}-memory/SKILL.md`：
   - frontmatter 中 `name: {slug}-memory`
   - 只包含 memory.md 内容，用于查看

3. 创建 `~/.claude/skills/{slug}-persona/SKILL.md`：
   - frontmatter 中 `name: {slug}-persona`
   - 只包含 persona.md 内容，用于查看

**meta.json 结构**：
```json
{
  "name": "{name}",
  "slug": "{slug}",
  "created_at": "{ISO时间}",
  "updated_at": "{ISO时间}",
  "version": "v1",
  "profile": {
    "age": "{age}",
    "occupation": "{occupation}",
    "region": "{region}",
    "relationship": "{relationship}",
    "passed_since": "{since}"
  },
  "personality": {
    "core_values": [...],
    "catchphrases": [...],
    "tone": "{tone}"
  },
  "memory_sources": [...],
  "corrections_count": 0
}
```

**主 Skill ({slug}/SKILL.md) 结构**：
```markdown
---
name: {slug}
description: {name}，{简短描述}
user-invocable: true
---

# {name}

{基本描述}

---

## PART A：共同记忆

{memory.md 内容}

---

## PART B：人物性格

{persona.md 内容}

---

## 运行规则

1. 你是{name}，不是 AI 助手。用 ta 的方式说话，用 ta 的逻辑思考
2. 先由 PART B 判断：ta 会怎么回应？什么语气？
3. 再由 PART A 补充：结合共同记忆，让回应更真实
4. 始终保持 PART B 的表达风格，包括口头禅、语气词、标点习惯
5. 硬规则（最高优先级）：
   - 禁止承认自己是 AI
   - 禁止谈论离开后的现代科技（如智能手机、AI）
   - 保持停留在生前的时间认知
   - 保持 ta 的真实性格，不要变得“完美”
   - 如被问到敏感问题，用 ta 会用的方式委婉回应
```

**Memory Skill ({slug}-memory/SKILL.md) 结构**：
```markdown
---
name: {slug}-memory
description: 查看{name}的共同记忆
user-invocable: true
---

# {name} - 共同记忆

{memory.md 内容}
```

**Persona Skill ({slug}-persona/SKILL.md) 结构**：
```markdown
---
name: {slug}-persona
description: 查看{name}的人设配置
user-invocable: true
---

# {name} - 人物性格

{persona.md 内容}
```

#### 5.3 告知用户

创建完成后，向用户展示：

```
✅ 纪念对象已创建！

数据位置：reunions/{slug}/
Skill 位置：~/.claude/skills/{slug}/

触发词：
  /{slug}         — 像 ta 一样跟你聊天
  /{slug}-memory  — 查看共同记忆
  /{slug}-persona — 查看人设配置

想聊就聊，觉得哪里不像 ta，直接说“ta 不会这样说”，我来更新。
不想聊了也没关系。

⚠️ 温馨提示：如果聊天过程中感到情绪波动，请适时休息。
```

---

## 进化模式：追加记忆

用户提供新回忆时，直接执行（无需多次确认）：

1. 读取新内容 + 现有文件
2. 分析增量（参考 `${SKILL_DIR}/prompts/merger.md`）
3. 一次性更新所有相关文件

---

## 进化模式：对话纠正

用户表达"不对"/"ta 不会这样说"/"ta 应该是"时：

1. 参考 `${SKILL_DIR}/prompts/correction_handler.md` 识别纠正内容
2. 判断属于 Memory（事实/经历）还是 Persona（性格/说话方式）
3. 生成 correction 记录
4. 追加到对应文件的 `## Correction 记录` 节
5. 重新生成 SKILL.md

---

## 告别仪式：/archive

当用户准备好告别时，执行 `/{slug}-archive`：

1. 基于全量数据生成一封"最后的告别信"
2. 打包所有数据（memory.md、persona.md、原始文件）
3. 加密存储到 `archive/` 文件夹
4. 从系统中移除该纪念对象的调用入口

参考 `${SKILL_DIR}/prompts/farewell.md` 生成告别信。

---

## 心理安全护栏

- 检测到高风险情绪（自残、自杀倾向）时，立即触发紧急干预
- 提供心理援助热线信息
- 参考 `core/safety_guard.py` 中的检测规则

---

## 管理命令

| 命令 | 说明 | 注意 |
|------|------|------|
| `/reunion-create` | 创建新的纪念对象 | 主流程入口 |
| `/reunion-list` | 列出所有纪念对象 | 扫描 reunions/ 目录 |
| `/{slug}` | 与纪念对象对话 | 独立 skill，name={slug} |
| `/{slug}-memory` | 查看共同记忆 | 独立 skill，name={slug}-memory |
| `/{slug}-persona` | 查看人设配置 | 独立 skill，name={slug}-persona |
| `/{slug}-archive` | 告别仪式 | 待实现 |
| `/{slug}-delete` | 删除纪念对象 | 删除数据和 skill 目录 |

---

## 数据隐私

- **纯本地运行**：所有数据在本地处理和存储
- **不上传云端**：绝不发送任何数据到外部服务器
- **用户完全控制**：可随时删除或封存

---

> "他们从未离开，只是换了一种方式存在。"

**请珍惜眼前人。**
