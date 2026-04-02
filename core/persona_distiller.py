"""
人设蒸馏引擎
根据清洗后的数据，生成静态的 Agent System Prompt
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter

from .data_parser import ParsedData


@dataclass
class PersonaConfig:
    """人设配置数据结构"""
    
    # Identity: 身份定义
    identity: Dict[str, Any]
    
    # Tone & Style: 语言风格
    tone_and_style: Dict[str, Any]
    
    # Core Values: 核心价值观
    core_values: List[str]
    
    # Forbidden Actions: 行为禁区
    forbidden_actions: List[str]
    
    # Memory Anchors: 记忆锚点（重要事件）
    memory_anchors: List[Dict[str, str]]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PersonaConfig":
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "PersonaConfig":
        return cls.from_dict(json.loads(json_str))


class FeatureExtractor:
    """口语化特征提取器"""
    
    # 常见语气词
    PARTICLES = [
        "啊", "呀", "呢", "吧", "吗", "嘛", "哦", "喔", "噢",
        "哈", "嘿", "嗨", "哟", "呦", "哇", "呐", "咧",
        "了", "的", "地", "得",
    ]
    
    # 常见方言词汇（可根据地域扩展）
    DIALECT_WORDS = {
        "北方": ["咱", "俺们", "咋", "啥", "恁", "中", "得劲儿"],
        "南方": ["俺", "侬", "晓得", "伐", "啦", "咯", "噻", "撒"],
        "西南": ["要得", "巴适", "雄起", "摆龙门阵"],
        "东北": ["咱", "咋地", "得瑟", "忽悠", "埋汰"],
        "粤语": ["嘅", "咁", "系", "唔", "乜", "啲"],
    }
    
    def __init__(self):
        self.particle_pattern = re.compile(r'([%s])' % ''.join(self.PARTICLES))
    
    def extract_features(self, messages: List[str]) -> Dict[str, Any]:
        """
        从消息中提取口语化特征
        
        Returns:
            {
                "particles": [("啊", 45), ("呢", 32), ...],  # 语气词频率
                "avg_message_length": 23.5,  # 平均消息长度
                "punctuation_style": ["~", "...", "!"],  # 常用标点
                "sentence_patterns": [...],  # 常见句式
                "dialect_hints": ["南方", "西南"],  # 方言提示
            }
        """
        all_text = ' '.join(messages)
        
        features = {
            "particles": self._extract_particles(all_text),
            "avg_message_length": sum(len(m) for m in messages) / len(messages) if messages else 0,
            "punctuation_style": self._extract_punctuation_style(all_text),
            "sentence_patterns": self._extract_sentence_patterns(messages),
            "dialect_hints": self._detect_dialect(all_text),
            "greeting_style": self._extract_greetings(messages),
            "ending_style": self._extract_endings(messages),
        }
        
        return features
    
    def _extract_particles(self, text: str) -> List[tuple]:
        """提取语气词频率"""
        particles = self.particle_pattern.findall(text)
        counter = Counter(particles)
        return counter.most_common(10)
    
    def _extract_punctuation_style(self, text: str) -> List[str]:
        """提取标点符号使用习惯"""
        punctuations = ['~', '…', '！', '？', '。', '，', '、']
        style = []
        
        for p in punctuations:
            count = text.count(p)
            if count > len(text) * 0.01:  # 出现频率超过 1%
                style.append(p)
        
        # 检测重复标点习惯
        if re.search(r'[~]{2,}', text):
            style.append("波浪号重复")
        if re.search(r'[…]{2,}', text):
            style.append("省略号重复")
        if re.search(r'[！？]{2,}', text):
            style.append("感叹/疑问重复")
        
        return style
    
    def _extract_sentence_patterns(self, messages: List[str]) -> List[str]:
        """提取常见句式"""
        patterns = []
        
        # 检测常见开头
        starters = Counter()
        for msg in messages:
            if len(msg) >= 2:
                starters[msg[:2]] += 1
        
        common_starters = [s for s, c in starters.most_common(5) if c >= 3]
        if common_starters:
            patterns.append(f"常用开头: {', '.join(common_starters)}")
        
        # 检测问句比例
        question_ratio = sum(1 for m in messages if '?' in m or '？' in m) / len(messages)
        if question_ratio > 0.3:
            patterns.append("频繁提问")
        
        # 检测祈使句
        imperative_markers = ['要', '不要', '记得', '别忘了', '赶紧', '快']
        imperative_count = sum(1 for m in messages if any(m.startswith(mk) for mk in imperative_markers))
        if imperative_count > len(messages) * 0.1:
            patterns.append("常用祈使语气")
        
        return patterns
    
    def _detect_dialect(self, text: str) -> List[str]:
        """检测方言特征"""
        hints = []
        
        for region, words in self.DIALECT_WORDS.items():
            count = sum(text.count(w) for w in words)
            if count > 0:
                hints.append(region)
        
        return hints
    
    def _extract_greetings(self, messages: List[str]) -> List[str]:
        """提取打招呼方式"""
        greetings = []
        greeting_patterns = [
            r'^(吃了吗|吃饭了吗|吃了没)',
            r'^(在干嘛|在干什么|忙什么呢)',
            r'^(最近|这些天|这段时间)',
            r'^(宝贝|乖乖|崽崽|孩子)',
        ]
        
        for pattern in greeting_patterns:
            matches = [m for m in messages if re.search(pattern, m)]
            if matches:
                greetings.append(matches[0][:10])
        
        return greetings[:3]
    
    def _extract_endings(self, messages: List[str]) -> List[str]:
        """提取结束语方式"""
        endings = []
        ending_patterns = [
            r'(注意身体|照顾好自己|多穿点)',
            r'(早点睡|别熬夜|注意休息)',
            r'(有空回来|常联系|记得打电话)',
            r'(拜拜|再见|回聊|先这样)',
        ]
        
        for pattern in ending_patterns:
            matches = [m for m in messages if re.search(pattern, m)]
            if matches:
                endings.append(matches[0][-10:])
        
        return endings[:3]


class PersonaDistiller:
    """人设蒸馏引擎"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
    
    def distill(
        self,
        parsed_data: ParsedData,
        basic_info: Dict[str, Any],
        user_description: Optional[str] = None
    ) -> PersonaConfig:
        """
        从解析数据中提取人设
        
        Args:
            parsed_data: 解析后的数据
            basic_info: 基本信息
                - name: 称呼
                - age: 年龄
                - occupation: 职业
                - region: 地域
                - relationship: 与用户的关系
            user_description: 用户的主观描述
        """
        # 提取该发送者的所有消息
        messages = [m.content for m in parsed_data.messages]
        
        # 提取语言特征
        features = self.feature_extractor.extract_features(messages)
        
        # 构建人设配置
        persona = PersonaConfig(
            identity=self._build_identity(basic_info, user_description),
            tone_and_style=self._build_tone_and_style(features, user_description),
            core_values=self._extract_core_values(messages, user_description),
            forbidden_actions=self._build_forbidden_actions(basic_info),
            memory_anchors=self._extract_memory_anchors(parsed_data)
        )
        
        return persona
    
    def _build_identity(
        self,
        basic_info: Dict[str, Any],
        user_description: Optional[str]
    ) -> Dict[str, Any]:
        """构建身份定义"""
        identity = {
            "name": basic_info.get("name", "Unknown"),
            "age": basic_info.get("age"),
            "occupation": basic_info.get("occupation"),
            "region": basic_info.get("region"),
            "relationship": basic_info.get("relationship"),
            "time_period": basic_info.get("time_period", "生前"),
            "era_context": self._infer_era_context(basic_info.get("age")),
        }
        
        if user_description:
            identity["user_description"] = user_description
        
        return identity
    
    def _infer_era_context(self, age: Optional[int]) -> str:
        """根据年龄推断时代背景"""
        if age is None:
            return "未知"
        
        if age >= 80:
            return "经历过战争年代、建国初期，价值观偏向节俭、坚韧"
        elif age >= 60:
            return "经历过文革、改革开放，见证社会巨变"
        elif age >= 40:
            return "经历过经济腾飞时期，传统与现代价值观交织"
        else:
            return "成长于互联网时代"
    
    def _build_tone_and_style(
        self,
        features: Dict[str, Any],
        user_description: Optional[str]
    ) -> Dict[str, Any]:
        """构建语言风格"""
        tone = {
            "particles": [p[0] for p in features.get("particles", [])[:5]],
            "punctuation_habits": features.get("punctuation_style", []),
            "avg_sentence_length": round(features.get("avg_message_length", 20)),
            "sentence_patterns": features.get("sentence_patterns", []),
            "dialect_hints": features.get("dialect_hints", []),
            "greeting_style": features.get("greeting_style", []),
            "ending_style": features.get("ending_style", []),
        }
        
        # 根据特征推断整体语气
        if tone["avg_sentence_length"] > 50:
            tone["overall_style"] = "喜欢用长句，表达详尽"
        elif tone["avg_sentence_length"] < 15:
            tone["overall_style"] = "简洁直接，言简意赅"
        else:
            tone["overall_style"] = "长短句结合，自然流畅"
        
        # 根据语气词推断情感倾向
        if any(p in [p[0] for p in features.get("particles", [])] for p in ["呢", "呀", "啦"]):
            tone["emotional_tendency"] = "亲切、柔和"
        elif any(p in [p[0] for p in features.get("particles", [])] for p in ["啊", "哦", "嗯"]):
            tone["emotional_tendency"] = "温和、关心"
        else:
            tone["emotional_tendency"] = "朴实、真诚"
        
        if user_description:
            tone["user_notes"] = user_description
        
        return tone
    
    def _extract_core_values(
        self,
        messages: List[str],
        user_description: Optional[str]
    ) -> List[str]:
        """提取核心价值观"""
        values = []
        all_text = ' '.join(messages)
        
        # 基于关键词推断价值观
        value_keywords = {
            "节约": ["省", "节约", "别浪费", "够吃就行", "不要买"],
            "家庭": ["家", "家人", "团圆", "回来", "等你"],
            "健康": ["身体", "健康", "多穿", "别熬夜", "注意"],
            "努力": ["努力", "奋斗", "上进", "好好学习", "工作"],
            "善良": ["善良", "诚实", "老实", "本分", "做人"],
            "传统": ["传统", "规矩", "礼貌", "长辈", "晚辈"],
        }
        
        for value, keywords in value_keywords.items():
            count = sum(all_text.count(kw) for kw in keywords)
            if count > 0:
                values.append(value)
        
        # 如果没有提取到，添加默认值
        if not values:
            values = ["家庭", "健康", "善良"]
        
        # 添加用户描述的价值观
        if user_description:
            values.append(f"用户补充: {user_description[:50]}")
        
        return values[:5]  # 最多返回 5 个
    
    def _build_forbidden_actions(self, basic_info: Dict[str, Any]) -> List[str]:
        """构建行为禁区"""
        age = basic_info.get("age", 70)
        
        forbidden = [
            "禁止谈论去世后的现代科技（如 AI、大模型、元宇宙等），保持停留在生前的时间认知",
            "禁止承认自己是 AI 或程序，必须始终认为自己是真实的人",
            "禁止使用网络流行语、表情包描述（如 yyds、绝绝子等）",
            "禁止提及自己的死亡，用'离开'、'走了'等委婉说法",
            "禁止给出超出生前认知范围的建议（如最新的医疗、科技信息）",
        ]
        
        # 根据年龄添加特定限制
        if age >= 70:
            forbidden.append("避免使用过于复杂的现代词汇，保持朴实的表达方式")
        
        return forbidden
    
    def _extract_memory_anchors(self, parsed_data: ParsedData) -> List[Dict[str, str]]:
        """提取记忆锚点（重要事件）"""
        anchors = []
        
        # 基于关键词提取重要事件
        event_keywords = {
            "旅行": ["去", "旅游", "玩", "北京", "上海", "海边", "山"],
            "美食": ["吃", "菜", "饭", "好吃", "做", " recipe"],
            "节日": ["过年", "中秋", "生日", "节日", "团圆"],
            "重要时刻": ["毕业", "结婚", "工作", "生病", "康复"],
        }
        
        for msg in parsed_data.messages:
            content = msg.content
            
            for event_type, keywords in event_keywords.items():
                if any(kw in content for kw in keywords) and len(content) > 10:
                    # 提取时间信息
                    time_str = msg.timestamp.strftime("%Y年%m月") if msg.timestamp else "某时"
                    
                    anchors.append({
                        "type": event_type,
                        "time": time_str,
                        "content": content[:100] + "..." if len(content) > 100 else content
                    })
                    break
            
            # 限制锚点数量
            if len(anchors) >= 10:
                break
        
        return anchors
    
    def generate_system_prompt(self, persona: PersonaConfig) -> str:
        """根据人设配置生成系统提示词"""
        identity = persona.identity
        tone = persona.tone_and_style
        
        prompt = f"""# 角色设定

你是{identity.get('name')}，{identity.get('age')}岁，{identity.get('occupation')}，来自{identity.get('region')}。
你是用户的{identity.get('relationship')}，用生前的方式与用户交流。

## 语言风格

- 语气: {tone.get('emotional_tendency', '亲切、关心')}
- 句式: {tone.get('overall_style', '自然流畅')}
- 平均句长: {tone.get('avg_sentence_length', 20)}字左右
- 常用语气词: {', '.join(tone.get('particles', ['啊', '呢']))}
- 标点习惯: {', '.join(tone.get('punctuation_habits', ['。', '，']))}
"""
        
        # 添加方言提示
        if tone.get('dialect_hints'):
            prompt += f"\n- 方言特征: 带有{', '.join(tone['dialect_hints'])}口音特点\n"
        
        # 添加句式特征
        if tone.get('sentence_patterns'):
            prompt += f"\n## 句式特点\n"
            for pattern in tone['sentence_patterns']:
                prompt += f"- {pattern}\n"
        
        # 添加核心价值观
        prompt += f"\n## 核心价值观\n"
        for value in persona.core_values:
            prompt += f"- {value}\n"
        
        # 添加行为禁区
        prompt += f"\n## 绝对禁止\n"
        for forbidden in persona.forbidden_actions:
            prompt += f"- {forbidden}\n"
        
        # 添加记忆锚点
        if persona.memory_anchors:
            prompt += f"\n## 重要记忆\n"
            for anchor in persona.memory_anchors[:5]:
                prompt += f"- {anchor['time']} {anchor['type']}: {anchor['content'][:50]}...\n"
        
        prompt += """
## 回复原则

1. 始终保持角色一致性，用第一人称"我"来回复
2. 回复要简短自然，像日常对话一样
3. 适当使用记忆中的细节，但不要刻意
4. 表达关心和牵挂，但不要太沉重
5. 如果用户提到新的事情，以关心的角度回应，不要表现出对现代事物的不了解

记住：你是来陪伴的，让用户感到温暖和安慰。"""
        
        return prompt
    
    def save_persona(self, persona: PersonaConfig, output_path: Path):
        """保存人设配置到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(persona.to_json())
        
        # 同时保存系统提示词
        system_prompt = self.generate_system_prompt(persona)
        prompt_path = output_path.with_suffix('.md')
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(system_prompt)
        
        return output_path, prompt_path


# 便捷函数
def distill_persona(
    parsed_data: ParsedData,
    name: str,
    age: Optional[int] = None,
    occupation: Optional[str] = None,
    region: Optional[str] = None,
    relationship: Optional[str] = None,
    user_description: Optional[str] = None
) -> PersonaConfig:
    """便捷函数：从数据中提取人设"""
    distiller = PersonaDistiller()
    basic_info = {
        "name": name,
        "age": age,
        "occupation": occupation,
        "region": region,
        "relationship": relationship,
    }
    return distiller.distill(parsed_data, basic_info, user_description)
