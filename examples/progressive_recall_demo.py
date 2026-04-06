"""
渐进式回忆功能演示

展示 ContextualMemoryEngine 如何使用
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.context_memory import ContextualMemoryEngine, ConversationContext


def demo_progressive_recall():
    """演示渐进式回忆效果"""
    
    # 模拟 memory.md 内容
    sample_memory = """
# 猫猫 - 共同记忆

## 关系概述

猫猫是一只橘猫，陪伴了主人5年。

---

## 关键记忆

### 1. 早上起床互动
- **时间**：2023年1月15日早上
- **地点**：家里
- **内容**：主人叫猫猫起床，猫猫伸懒腰翻身继续睡，被主人笑称"小懒猫"
- **感受**：日常的温馨互动，展现了猫猫的慵懒可爱

### 2. 下班回家迎接
- **时间**：2023年3月20日
- **地点**：家里窗台
- **内容**：主人下班回家，猫猫从窗台跳下来迎接，用头蹭主人的手
- **感受**：每天回家的温暖时刻，猫猫的等待让主人感到被需要

### 3. 生病时的陪伴
- **时间**：2023年11月20日
- **地点**：床上
- **内容**：主人生病不舒服，猫猫跳上床趴在身边，轻轻用爪子碰主人的手
- **感受**：猫猫的暖心陪伴，即使不会说话也能感知主人的情绪

### 4. 中秋看月亮
- **时间**：2023年9月10日中秋节
- **地点**：窗边
- **内容**：主人和猫猫一起看月亮，主人说月亮像猫猫的眼睛
- **感受**：浪漫的共享时刻，主人承诺每年中秋都陪猫猫看月亮

### 5. 窗边拍照时刻
- **时间**：2024年3月15日下午
- **地点**：窗边
- **内容**：猫猫坐在窗边，主人夸ta漂亮并拍照，猫猫摆出完美pose
- **感受**：美好的日常记录，猫猫仿佛懂得配合拍照

---

## 日常点滴

- 猫猫喜欢坐在窗边看风景，一坐就是很久
- 主人回家时，猫猫总会从窗台跳下来迎接
- 猫猫是傲娇小吃货，听到开罐头的声音就冲过来
- 喜欢玩逗猫棒，玩累了就趴下翻肚皮
- 经常喵喵叫回应主人的话，像个话痨小伙伴
- 开心时会发出呼噜呼噜的声音，表示舒服和满足
- 被摸下巴时会特别享受

---

## 饮食记忆

### ta 喜欢的食物
- **猫粮**：听到倒猫粮的声音就兴奋跑过来
- **罐头**：开罐头的"咔嚓"声是猫猫的最爱
- **冰淇淋**：夏天时会给猫猫舔一口冰淇淋，ta很喜欢

---

## 温馨瞬间

1. **生病陪伴**：猫猫感知到主人不舒服，主动跳上床陪伴
2. **新年跳腿**：2024年新年，猫猫跳到主人腿上，虽然被说"好重"但依然呼噜呼噜
3. **蹭手回应**：每次主人问"想我了吗？"，猫猫就用头蹭手回应
4. **默契配合**：拍照时猫猫会摆出完美pose，仿佛懂得主人的心意
"""
    
    # 创建临时文件
    temp_file = Path("temp_memory.md")
    temp_file.write_text(sample_memory, encoding='utf-8')
    
    # 初始化引擎
    engine = ContextualMemoryEngine(temp_file)
    
    print("=" * 60)
    print("渐进式回忆系统演示")
    print("=" * 60)
    print(f"\n已加载 {len(engine.memories)} 条记忆\n")
    
    # 模拟对话场景
    test_inputs = [
        "今天天气真好",
        "我在窗边晒太阳",
        "最近工作好累",
        "你饿了吗",
        "想你了",
        "记得我们以前一起看月亮吗",
        "今天生病了",
        "你在干嘛",
    ]
    
    print("-" * 60)
    print("开始模拟对话:\n")
    
    for user_input in test_inputs:
        print(f"用户: {user_input}")
        
        # 处理输入
        memory = engine.process_user_input(user_input)
        
        if memory:
            hint = engine.get_memory_hint(memory)
            print(f"猫猫: {hint}")
            print(f"     [激活度: {memory.activation:.2f}, 关键词: {', '.join(memory.keywords[:3])}]")
        else:
            print("猫猫: 喵~（简单回应，没有特定记忆被激活）")
        
        print()
    
    # 显示最终状态
    print("-" * 60)
    print("\n对话结束后的记忆状态:")
    print(engine.get_context_prompt())
    
    # 清理
    temp_file.unlink()


def demo_emotion_synergy():
    """演示情感共鸣效果"""
    
    print("\n" + "=" * 60)
    print("情感共鸣演示")
    print("=" * 60)
    
    # 模拟不同情绪下的记忆激活
    scenarios = [
        ("今天好开心", "happy"),
        ("最近好难过", "sad"),
        ("一个人好孤单", "lonely"),
        ("觉得好累", "tired"),
    ]
    
    for text, expected_emotion in scenarios:
        print(f"\n用户情绪: {expected_emotion}")
        print(f"用户说: {text}")
        print("-> 系统会优先唤起温暖/治愈相关的记忆")


if __name__ == "__main__":
    demo_progressive_recall()
    demo_emotion_synergy()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
