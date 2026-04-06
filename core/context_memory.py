"""
渐进式回忆系统 (Context-Aware Memory Retrieval)

核心概念：
- 记忆不是一次性暴露，而是根据对话上下文动态激活
- 每条记忆有"激活度"，只有达到一定阈值才会被提及
- 模拟真实人类：突然想起某件事，或顺着话题自然回忆
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import random


@dataclass
class MemoryItem:
    """单个记忆条目"""
    id: str
    content: str  # 记忆内容
    keywords: List[str]  # 关键词标签
    emotion_tags: List[str]  # 情感标签: happy, sad, warm, funny...
    importance: int  # 重要程度 1-5
    activation: float = 0.0  # 当前激活度 0-1
    last_mentioned: Optional[datetime] = None  # 上次提及时间
    mention_count: int = 0  # 被提及次数
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "keywords": self.keywords,
            "emotion_tags": self.emotion_tags,
            "importance": self.importance,
            "activation": self.activation,
            "last_mentioned": self.last_mentioned.isoformat() if self.last_mentioned else None,
            "mention_count": self.mention_count
        }


@dataclass
class ConversationContext:
    """对话上下文"""
    recent_topics: List[str] = field(default_factory=list)  # 最近话题
    mentioned_memories: Set[str] = field(default_factory=set)  # 已提及的记忆ID
    user_emotion: str = "neutral"  # 用户当前情绪
    conversation_turn: int = 0  # 对话轮数
    
    def add_topic(self, topic: str):
        """添加话题，保持最近5个"""
        self.recent_topics.append(topic)
        if len(self.recent_topics) > 5:
            self.recent_topics.pop(0)
    
    def mark_memory_mentioned(self, memory_id: str):
        """标记记忆已被提及"""
        self.mentioned_memories.add(memory_id)


class ContextualMemoryEngine:
    """
    渐进式回忆引擎
    
    核心算法：
    1. 关键词匹配 → 初步筛选相关记忆
    2. 激活度计算 → 考虑重要性、情感共鸣、时间衰减
    3. 自然涌现 → 模拟"突然想起"的效果
    """
    
    # 情感共鸣权重
    EMOTION_SYNERGY = {
        ("happy", "happy"): 1.2,
        ("sad", "sad"): 1.5,  # 悲伤时更容易想起悲伤的事
        ("sad", "warm"): 1.3,  # 悲伤时温暖回忆有治愈效果
        ("lonely", "warm"): 1.4,
        ("excited", "funny"): 1.2,
    }
    
    def __init__(self, memory_file: Optional[Path] = None):
        self.memories: Dict[str, MemoryItem] = {}
        self.context = ConversationContext()
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)  # 关键词 -> 记忆ID
        
        if memory_file and memory_file.exists():
            self.load_memories(memory_file)
    
    def load_memories(self, memory_file: Path):
        """从 memory.md 解析记忆"""
        content = memory_file.read_text(encoding='utf-8')
        self._parse_memory_content(content)
    
    def _parse_memory_content(self, content: str):
        """解析记忆内容，提取结构化数据"""
        # 按章节分割
        sections = re.split(r'\n##+\s+', content)
        
        memory_id = 0
        for section in sections:
            if not section.strip():
                continue
            
            # 提取记忆条目（以 ### 开头）
            memories = re.findall(
                r'###\s+(.+?)\n- \*\*时间\*\*：(.+?)\n- \*\*内容\*\*：(.+?)(?=\n###|\n##|$)',
                section,
                re.DOTALL
            )
            
            for title, time, content in memories:
                memory_id += 1
                item = MemoryItem(
                    id=f"mem_{memory_id:03d}",
                    content=content.strip(),
                    keywords=self._extract_keywords(title + " " + content),
                    emotion_tags=self._detect_emotion(content),
                    importance=self._assess_importance(content)
                )
                self.memories[item.id] = item
                
                # 建立关键词索引
                for kw in item.keywords:
                    self.keyword_index[kw].append(item.id)
        
        # 如果没有解析到结构化数据，尝试简单分割
        if not self.memories:
            self._parse_simple_content(content)
    
    def _parse_simple_content(self, content: str):
        """简单解析：按段落分割"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, para in enumerate(paragraphs[:20]):  # 最多20条
            item = MemoryItem(
                id=f"mem_{i+1:03d}",
                content=para[:200],  # 限制长度
                keywords=self._extract_keywords(para),
                emotion_tags=self._detect_emotion(para),
                importance=random.randint(2, 4)
            )
            self.memories[item.id] = item
            for kw in item.keywords:
                self.keyword_index[kw].append(item.id)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词 - 增强版"""
        keywords = []
        
        # 1. 场景关键词（高优先级）
        scene_keywords = [
            '窗边', '晒太阳', '睡觉', '起床', '回家', '下班',
            '生病', '病了', '中秋', '月亮', '拍照', '馆头',
            '饭', '吃饭', '饭粥', '工作', '累', '休息',
            '想你', '想念', '抱抱', '陪伴', '睡觉', '睡'
        ]
        for kw in scene_keywords:
            if kw in text:
                keywords.append(kw)
        
        # 2. 提取两字词
        words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
        stopwords = {'我们', '他们', '这个', '那个', '什么', '怎么', 
                     '没有', '可以', '就是', '一个', '自己', '这样',
                     '那么', '怎么', '还是', '不是', '就是'}
        words = [w for w in words if w not in stopwords]
        keywords.extend(words)
        
        # 去重并返回
        return list(dict.fromkeys(keywords))[:8]  # 保持顺序去重，最多8个
    
    def _detect_emotion(self, text: str) -> List[str]:
        """检测情感标签"""
        emotions = []
        
        emotion_keywords = {
            "happy": ["开心", "高兴", "笑", "快乐", "好玩", "有趣"],
            "sad": ["难过", "伤心", "哭", "遗憾", "想念", "舍不得"],
            "warm": ["温暖", "陪伴", "关心", "照顾", "爱", "谢谢"],
            "funny": ["好笑", "逗", "傻", "调皮", "搞怪"],
            "lonely": ["孤单", "寂寞", "一个人", "等你", "想你"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(kw in text for kw in keywords):
                emotions.append(emotion)
        
        return emotions if emotions else ["neutral"]
    
    def _assess_importance(self, content: str) -> int:
        """评估重要性 1-5"""
        score = 3  # 基础分
        
        # 长度因素
        if len(content) > 100:
            score += 1
        
        # 情感浓度
        emotion_words = len(self._detect_emotion(content))
        score += min(emotion_words, 1)
        
        # 特殊标记
        important_markers = ['第一次', '最后', '永远', '最重要', '最难忘']
        if any(m in content for m in important_markers):
            score += 1
        
        return min(max(score, 1), 5)
    
    def process_user_input(self, user_input: str) -> Optional[MemoryItem]:
        """
        处理用户输入，返回应该被激活的记忆
        
        返回 None 表示没有足够相关的记忆被激活
        """
        self.context.conversation_turn += 1
        
        # 1. 提取当前话题
        current_topics = self._extract_keywords(user_input)
        for topic in current_topics:
            self.context.add_topic(topic)
        
        # 2. 更新所有记忆的激活度
        self._update_activation(user_input)
        
        # 3. 选择最相关的记忆
        activated_memory = self._select_memory()
        
        if activated_memory:
            # 标记已提及
            self.context.mark_memory_mentioned(activated_memory.id)
            activated_memory.last_mentioned = datetime.now()
            activated_memory.mention_count += 1
            # 重置激活度
            activated_memory.activation = 0.0
        
        return activated_memory
    
    def _update_activation(self, user_input: str):
        """更新记忆激活度"""
        user_keywords = set(self._extract_keywords(user_input))
        user_emotions = set(self._detect_emotion(user_input))
        
        for memory_id, memory in self.memories.items():
            # 跳过已频繁提及的记忆（避免重复）
            if memory.mention_count >= 3:
                continue
            
            # 基础激活：关键词匹配
            keyword_match = len(set(memory.keywords) & user_keywords)
            base_activation = keyword_match * 0.2
            
            # 话题连续性：与最近话题相关
            topic_bonus = 0
            for topic in self.context.recent_topics:
                if topic in memory.keywords:
                    topic_bonus += 0.1
            
            # 情感共鸣
            emotion_bonus = 0
            for ue in user_emotions:
                for me in memory.emotion_tags:
                    synergy = self.EMOTION_SYNERGY.get((ue, me), 1.0)
                    emotion_bonus += (synergy - 1.0) * 0.2
            
            # 重要性权重
            importance_weight = memory.importance * 0.05
            
            # 时间衰减：最近提过的记忆暂时降低激活度
            time_decay = 1.0
            if memory.last_mentioned:
                hours_since = (datetime.now() - memory.last_mentioned).total_seconds() / 3600
                if hours_since < 1:
                    time_decay = 0.3  # 1小时内大幅降低
                elif hours_since < 24:
                    time_decay = 0.7
            
            # 随机因素：模拟"突然想起"
            random_spark = random.uniform(0, 0.1) if memory.importance >= 4 else 0
            
            # 综合计算
            new_activation = (
                base_activation +
                topic_bonus +
                emotion_bonus +
                importance_weight +
                random_spark
            ) * time_decay
            
            # 平滑更新
            memory.activation = memory.activation * 0.3 + new_activation * 0.7
    
    def _select_memory(self) -> Optional[MemoryItem]:
        """选择要激活的记忆"""
        # 阈值：只有激活度 > 0.5 的记忆才会被考虑
        candidates = [
            m for m in self.memories.values()
            if m.activation > 0.5 and m.id not in self.context.mentioned_memories
        ]
        
        if not candidates:
            # 如果没有新记忆，考虑已提及但很久没说的
            old_candidates = [
                m for m in self.memories.values()
                if m.activation > 0.3 and m.mention_count < 2
            ]
            candidates = old_candidates
        
        if not candidates:
            return None
        
        # 按激活度排序，但加入随机性
        candidates.sort(key=lambda m: m.activation, reverse=True)
        
        # 80%概率选最高，20%概率随机选（增加自然感）
        if random.random() < 0.8:
            return candidates[0]
        else:
            return random.choice(candidates[:3]) if len(candidates) >= 3 else candidates[0]
    
    def get_memory_hint(self, memory: MemoryItem) -> str:
        """
        生成记忆提示，用于引导AI如何自然提及这段记忆
        """
        templates = {
            "happy": [
                "突然想起{content}...",
                "记得吗？{content}",
                "说到这个，{content}",
            ],
            "sad": [
                "有时候会想起{content}...",
                "其实我一直记得{content}",
            ],
            "warm": [
                "那时候{content}，真好啊...",
                "{content}，我一直记在心里",
            ],
            "funny": [
                "哈哈，还记得{content}吗？",
                "想起来就好笑，{content}",
            ],
            "neutral": [
                "对了，{content}",
                "说到这个，{content}",
            ]
        }
        
        # 选择主要情感
        primary_emotion = memory.emotion_tags[0] if memory.emotion_tags else "neutral"
        template_pool = templates.get(primary_emotion, templates["neutral"])
        
        template = random.choice(template_pool)
        
        # 简化内容，避免太长
        short_content = memory.content[:50] + "..." if len(memory.content) > 50 else memory.content
        
        return template.format(content=short_content)
    
    def get_context_prompt(self) -> str:
        """
        生成给AI的上下文提示
        """
        lines = ["\n### 渐进式回忆系统状态"]
        lines.append(f"- 当前对话轮数: {self.context.conversation_turn}")
        lines.append(f"- 最近话题: {', '.join(self.context.recent_topics[-3:])}")
        lines.append(f"- 已提及记忆数: {len(self.context.mentioned_memories)}/{len(self.memories)}")
        
        # 高激活度记忆（供AI参考）
        high_activation = [
            m for m in self.memories.values()
            if m.activation > 0.3 and m.id not in self.context.mentioned_memories
        ]
        high_activation.sort(key=lambda m: m.activation, reverse=True)
        
        if high_activation:
            lines.append("\n**可能相关的记忆**（根据当前话题）:")
            for m in high_activation[:3]:
                lines.append(f"- [{m.activation:.2f}] {m.content[:40]}...")
        
        return "\n".join(lines)


# 便捷函数
def create_context_engine(memory_file: Path) -> ContextualMemoryEngine:
    """创建上下文记忆引擎"""
    return ContextualMemoryEngine(memory_file)


def simulate_progressive_recall(user_input: str, memory_file: Path) -> Optional[str]:
    """
    模拟渐进式回忆
    
    返回记忆提示，或 None
    """
    engine = ContextualMemoryEngine(memory_file)
    memory = engine.process_user_input(user_input)
    
    if memory:
        return engine.get_memory_hint(memory)
    
    return None
