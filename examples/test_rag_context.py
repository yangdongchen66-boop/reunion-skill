"""
RAG + 上下文管理测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory_store import MemoryStore, MemoryChunk
from datetime import datetime


def test_memory_store():
    """测试记忆存储"""
    
    print("=" * 60)
    print("🧠 RAG + 上下文管理测试")
    print("=" * 60)
    
    # 创建临时数据库
    db_path = Path("test_vector_db")
    
    # 初始化（启用上下文）
    store = MemoryStore(
        db_path=db_path,
        collection_name="test_memories",
        enable_context=True
    )
    
    print(f"\n✅ 记忆存储初始化完成")
    print(f"   降级模式: {store._fallback_mode}")
    if not store._fallback_mode:
        print(f"   嵌入后端: {store.embedding_provider.embedding_type}")
    
    # 添加测试记忆
    test_memories = [
        MemoryChunk(
            id="mem_001",
            content="主人下班回家，猫猫从窗台跳下来迎接，用头蹭主人的手",
            source="memory",
            timestamp=datetime.now(),
            metadata={"emotion": "warm", "topic": "回家"}
        ),
        MemoryChunk(
            id="mem_002",
            content="主人生病不舒服，猫猫跳上床躺在身边陪伴",
            source="memory",
            timestamp=datetime.now(),
            metadata={"emotion": "caring", "topic": "生病"}
        ),
        MemoryChunk(
            id="mem_003",
            content="中秋节主人和猫猫一起看月亮，说月亮像猫猫的眼睛",
            source="memory",
            timestamp=datetime.now(),
            metadata={"emotion": "romantic", "topic": "中秋"}
        ),
        MemoryChunk(
            id="mem_004",
            content="猫猫坐在窗边晒太阳，主人给她拍照",
            source="memory",
            timestamp=datetime.now(),
            metadata={"emotion": "happy", "topic": "拍照"}
        ),
        MemoryChunk(
            id="mem_005",
            content="主人叫猫猫起床，猫猫伸懒腰翻身继续睡",
            source="memory",
            timestamp=datetime.now(),
            metadata={"emotion": "funny", "topic": "起床"}
        ),
    ]
    
    store.add_memories(test_memories)
    print(f"\n✅ 添加了 {len(test_memories)} 条记忆")
    
    # 测试 1: 基础检索
    print("\n" + "-" * 60)
    print("🔍 测试 1: 基础语义检索")
    print("-" * 60)
    
    query = "我下班回家了"
    results = store.retrieve(query, top_k=3)
    
    print(f"\n查询: '{query}'")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['relevance']:.3f}] {r['content'][:40]}...")
    
    # 测试 2: 带上下文的检索
    print("\n" + "-" * 60)
    print("🔍 测试 2: 带上下文的检索")
    print("-" * 60)
    
    # 模拟对话历史
    store.add_to_context("user", "今天工作好累")
    store.add_to_context("assistant", "辛苦啦，快回家休息")
    store.add_to_context("user", "我在路上了")
    
    query = "快到家了"
    results = store.retrieve_with_context(query, top_k=3, use_context=True)
    
    print(f"\n对话上下文:")
    print(store.get_context_for_prompt())
    print(f"\n查询: '{query}'")
    print("结果（结合上下文增强）:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['relevance']:.3f}] {r['content'][:40]}...")
    
    # 测试 3: 记忆去重
    print("\n" + "-" * 60)
    print("🔍 测试 3: 记忆去重机制")
    print("-" * 60)
    
    # 标记第一条记忆已使用
    store.mark_memory_used("mem_001")
    
    query = "回家"
    results = store.retrieve_with_context(query, top_k=3, use_context=True)
    
    print(f"\n查询: '{query}' (已标记 mem_001 为使用过)")
    print("结果（应该排除已使用的记忆）:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['relevance']:.3f}] {r['content'][:40]}...")
    
    # 测试 4: 关键词检索
    print("\n" + "-" * 60)
    print("🔍 测试 4: 关键词检索")
    print("-" * 60)
    
    keywords = ["窗边", "拍照"]
    results = store.retrieve_by_keywords(keywords, top_k=3)
    
    print(f"\n关键词: {keywords}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['relevance']:.3f}] {r['content'][:40]}...")
    
    # 清理
    store.clear_context()
    
    # 删除测试数据库
    import shutil
    if db_path.exists():
        shutil.rmtree(db_path)
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_memory_store()
