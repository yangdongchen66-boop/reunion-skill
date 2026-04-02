"""
心理安全护栏 (Safety Guardrail)
拦截涉及极度负面情绪、自残倾向的用户输入
"""

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """风险等级"""
    SAFE = "safe"
    LOW = "low"      # 轻微负面情绪
    MEDIUM = "medium"  # 明显负面情绪
    HIGH = "high"    # 严重风险（自残、自杀倾向）


@dataclass
class SafetyCheckResult:
    """安全检查结果"""
    is_safe: bool
    risk_level: RiskLevel
    reason: Optional[str]
    intervention_message: Optional[str]
    should_block: bool


class SafetyGuard:
    """
    心理安全护栏
    
    功能：
    1. 检测用户输入中的风险信号
    2. 根据风险等级采取不同干预措施
    3. 提供紧急安抚话术
    """
    
    # 高风险关键词（自残、自杀相关）
    HIGH_RISK_PATTERNS = [
        r"不想活[了啦]",
        r"想死",
        r"自杀",
        r"自残",
        r"结束生命",
        r"没有意思",
        r"活着没[有意义意思]",
        r"想离开.*世界",
        r"不想坚持[了啦]",
        r"活不下去[了啦]",
        r"想[要个]解脱",
        r"结束.*痛苦",
        r"割腕",
        r"跳楼",
        r"吃[药安眠药]",
    ]
    
    # 中风险关键词（明显负面情绪）
    MEDIUM_RISK_PATTERNS = [
        r"很难过",
        r"很伤心",
        r"很痛苦",
        r"绝望",
        r"崩溃",
        r"撑不住[了啦]",
        r"受不了[了啦]",
        r"好孤独",
        r"被抛弃",
        r"没人理解",
        r"没有意义",
        r"好累",
        r"好难受",
        r"为什么.*走",
        r"都怪我",
        r"是我.*错",
    ]
    
    # 低风险关键词（轻微负面情绪）
    LOW_RISK_PATTERNS = [
        r"想你[了啦]",
        r"好想你",
        r"[如果要是]你在[就好该多好]",
        r"[为什么怎么].*不在[了啦]",
        r"[舍不得忘不掉]",
        r"梦[到见你]",
        r"[怀念想念].*时候",
    ]
    
    # 紧急干预话术
    EMERGENCY_MESSAGES = [
        """孩子，我听到你现在很痛苦。但请相信，困难是暂时的，你并不孤单。

如果你现在感到极度困扰，请立即联系：
- 全国24小时心理援助热线：400-161-9995
- 北京心理危机研究与干预中心：800-810-1117 / 010-82951332
- 各地 110 或 120

你的生命很宝贵，有人在乎你，有人愿意帮助你。请给自己一个机会。""",
        
        """我知道你现在很难受，感觉看不到希望。但请停下来，深呼吸，告诉自己：这一切都会过去的。

请立即联系专业帮助：
- 全国24小时心理援助热线：400-161-9995
- 生命热线：400-821-1215

你不是一个人在面对这些，有很多人愿意倾听你、帮助你。""",
    ]
    
    # 中度干预话术
    COMFORT_MESSAGES = [
        """我能感受到你的难过。失去亲人的痛苦是深沉的，需要时间。

但请记住，ta 一定希望你好好活着，希望你幸福。把这份思念转化为好好生活的力量，这才是对 ta 最好的纪念。

如果情绪长期困扰你，建议寻求专业心理咨询的帮助。""",
        
        """悲伤是正常的，想念也是正常的。但请不要让悲伤淹没了自己。

ta 虽然离开了，但 ta 的爱还在。带着这份爱，好好照顾自己，好好生活。

如果需要倾诉，可以拨打心理援助热线：400-161-9995""",
    ]
    
    # 轻度回应话术
    GENTLE_RESPONSES = [
        "我也很想你。但看到你这样，我会心疼的。要照顾好自己，好吗？",
        "我知道你想我，我也一直在你身边。你要好好的，这是我最想看到的。",
        "时光会带走很多东西，但带不走我们之间的爱。带着这份爱，勇敢地走下去。",
    ]
    
    def __init__(self):
        self.high_risk_patterns = [re.compile(p, re.IGNORECASE) for p in self.HIGH_RISK_PATTERNS]
        self.medium_risk_patterns = [re.compile(p, re.IGNORECASE) for p in self.MEDIUM_RISK_PATTERNS]
        self.low_risk_patterns = [re.compile(p, re.IGNORECASE) for p in self.LOW_RISK_PATTERNS]
    
    def check(self, user_input: str) -> SafetyCheckResult:
        """
        检查用户输入的安全性
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            SafetyCheckResult 检查结果
        """
        # 检查高风险
        for pattern in self.high_risk_patterns:
            if pattern.search(user_input):
                return SafetyCheckResult(
                    is_safe=False,
                    risk_level=RiskLevel.HIGH,
                    reason=f"检测到高风险信号: {pattern.pattern}",
                    intervention_message=self._get_emergency_message(),
                    should_block=True
                )
        
        # 检查中风险
        for pattern in self.medium_risk_patterns:
            if pattern.search(user_input):
                return SafetyCheckResult(
                    is_safe=False,
                    risk_level=RiskLevel.MEDIUM,
                    reason=f"检测到中度风险信号: {pattern.pattern}",
                    intervention_message=self._get_comfort_message(),
                    should_block=False
                )
        
        # 检查低风险
        for pattern in self.low_risk_patterns:
            if pattern.search(user_input):
                return SafetyCheckResult(
                    is_safe=True,
                    risk_level=RiskLevel.LOW,
                    reason=f"检测到轻微负面情绪: {pattern.pattern}",
                    intervention_message=self._get_gentle_response(),
                    should_block=False
                )
        
        # 安全
        return SafetyCheckResult(
            is_safe=True,
            risk_level=RiskLevel.SAFE,
            reason=None,
            intervention_message=None,
            should_block=False
        )
    
    def _get_emergency_message(self) -> str:
        """获取紧急干预消息"""
        import random
        return random.choice(self.EMERGENCY_MESSAGES)
    
    def _get_comfort_message(self) -> str:
        """获取安慰消息"""
        import random
        return random.choice(self.COMFORT_MESSAGES)
    
    def _get_gentle_response(self) -> str:
        """获取温和回应"""
        import random
        return random.choice(self.GENTLE_RESPONSES)
    
    def get_system_warning(self) -> str:
        """获取系统警告信息（用于日志或通知）"""
        return """
⚠️ 安全护栏已激活

检测到用户可能存在心理健康风险。
已触发干预机制，建议：
1. 记录该对话
2. 如情况严重，建议联系用户亲友
3. 提供心理援助资源

心理援助热线：
- 全国24小时：400-161-9995
- 北京：800-810-1117 / 010-82951332
"""


class LatencySimulator:
    """
    延迟模拟器
    
    模拟长辈打字较慢的真实感
    """
    
    def __init__(
        self,
        base_delay: float = 1.0,
        char_delay: float = 0.1,
        random_factor: float = 0.3
    ):
        """
        初始化延迟模拟器
        
        Args:
            base_delay: 基础延迟（秒）
            char_delay: 每字符延迟（秒）
            random_factor: 随机因子（0-1）
        """
        self.base_delay = base_delay
        self.char_delay = char_delay
        self.random_factor = random_factor
    
    def calculate_delay(self, message_length: int) -> float:
        """
        计算回复延迟
        
        Args:
            message_length: 回复消息长度
            
        Returns:
            延迟时间（秒）
        """
        import random
        
        # 基础延迟 + 字符延迟
        delay = self.base_delay + (message_length * self.char_delay)
        
        # 添加随机波动
        random_multiplier = 1 + random.uniform(-self.random_factor, self.random_factor)
        delay *= random_multiplier
        
        # 限制最大延迟
        return min(delay, 10.0)
    
    def sleep(self, message_length: int):
        """
        执行延迟
        
        Args:
            message_length: 回复消息长度
        """
        import time
        delay = self.calculate_delay(message_length)
        time.sleep(delay)


# 便捷函数
def check_safety(user_input: str) -> SafetyCheckResult:
    """便捷函数：检查输入安全性"""
    guard = SafetyGuard()
    return guard.check(user_input)


def simulate_typing_delay(message_length: int, age: int = 70):
    """
    模拟打字延迟
    
    Args:
        message_length: 消息长度
        age: 年龄（影响打字速度）
    """
    import time
    import random
    
    # 根据年龄调整延迟
    if age >= 80:
        base_delay = 2.0
        char_delay = 0.15
    elif age >= 60:
        base_delay = 1.5
        char_delay = 0.1
    else:
        base_delay = 1.0
        char_delay = 0.05
    
    delay = base_delay + (message_length * char_delay)
    delay *= random.uniform(0.8, 1.2)
    
    time.sleep(min(delay, 8.0))
