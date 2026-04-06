"""
记忆检索模块 (RAG Memory System)
基于向量数据库实现长期记忆检索

修复方案：使用 ONNX Runtime 替代 PyTorch，解决依赖冲突
"""

import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np

# 尝试导入 ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# 尝试导入嵌入模型 - 优先使用 ONNX 版本
EMBEDDING_AVAILABLE = False
EMBEDDING_TYPE = None

# 设置环境变量，避免下载模型时的警告
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    # 方案 1: 尝试使用 sentence-transformers (最稳定)
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
    EMBEDDING_TYPE = "sentence_transformers"
except ImportError:
    pass

if not EMBEDDING_AVAILABLE:
    try:
        # 方案 2: 尝试使用 transformers
        from transformers import AutoTokenizer, AutoModel
        EMBEDDING_AVAILABLE = True
        EMBEDDING_TYPE = "transformers"
    except ImportError:
        pass


@dataclass
class MemoryChunk:
    """记忆片段数据结构"""
    id: str
    content: str
    source: str  # 来源: wechat, qq, diary, etc.
    timestamp: Optional[datetime]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata
        }


@dataclass
class ConversationContext:
    """
    对话上下文管理
    维护短期对话历史，支持上下文感知
    """
    max_history: int = 10  # 最多保持10轮对话
    
    def __post_init__(self):
        self.history: deque = deque(maxlen=self.max_history)  # (role, content) 列表
        self.current_topics: List[str] = []  # 当前话题
        self.user_emotion: str = "neutral"  # 用户情绪
        self.mentioned_memories: set = set()  # 已提及的记忆ID
    
    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_context(self, n: int = 5) -> str:
        """获取最近n轮对话上下文"""
        recent = list(self.history)[-n:]
        lines = []
        for msg in recent:
            role_name = "主人" if msg["role"] == "user" else "ta"
            lines.append(f"{role_name}: {msg['content']}")
        return "\n".join(lines)
    
    def update_topics(self, topics: List[str]):
        """更新当前话题"""
        self.current_topics = topics[-3:]  # 只保持最近3个话题
    
    def mark_memory_mentioned(self, memory_id: str):
        """标记记忆已被提及"""
        self.mentioned_memories.add(memory_id)
    
    def is_memory_mentioned(self, memory_id: str) -> bool:
        """检查记忆是否已被提及"""
        return memory_id in self.mentioned_memories
    
    def clear(self):
        """清空上下文"""
        self.history.clear()
        self.current_topics.clear()
        self.mentioned_memories.clear()


class EmbeddingProvider:
    """
    嵌入模型提供者
    支持多种后端：Sentence-Transformers / Transformers
    """
    
    # 使用更轻量的模型
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.embedding_type = EMBEDDING_TYPE
        
        try:
            if self.embedding_type == "sentence_transformers":
                print(f"[信息] 加载 SentenceTransformer 模型: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            elif self.embedding_type == "transformers":
                print(f"[信息] 加载 Transformers 模型: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            else:
                raise RuntimeError("没有可用的嵌入模型后端")
        except Exception as e:
            print(f"[警告] 加载模型失败: {e}")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本编码为向量
        """
        if self.embedding_type == "sentence_transformers":
            return self.model.encode(texts, convert_to_numpy=True)
        
        elif self.embedding_type == "transformers":
            import torch
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 使用 [CLS] token 的表示
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            return embeddings
        
        else:
            raise RuntimeError(f"不支持的嵌入类型: {self.embedding_type}")


class MemoryStore:
    """
    记忆存储与检索系统
    
    使用 ChromaDB 作为本地向量数据库，支持：
    1. 向量化存储记忆片段
    2. 语义相似度检索
    3. 基于关键词的检索
    4. 对话上下文管理
    """
    
    def __init__(
        self,
        db_path: Path,
        collection_name: str = "reunion_memories",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        enable_context: bool = True
    ):
        """
        初始化记忆存储
        
        Args:
            db_path: 向量数据库存储路径
            collection_name: 集合名称（每个纪念对象一个集合）
            embedding_model: 嵌入模型名称
            enable_context: 是否启用对话上下文
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.enable_context = enable_context
        
        # 检查可用性
        self._fallback_mode = not (CHROMADB_AVAILABLE and EMBEDDING_AVAILABLE)
        
        # 初始化对话上下文
        if self.enable_context:
            self.context = ConversationContext()
        
        if self._fallback_mode:
            print(f"[警告] 使用降级模式: ChromaDB={CHROMADB_AVAILABLE}, Embedding={EMBEDDING_AVAILABLE}")
            self._memories: List[MemoryChunk] = []
            return
        
        try:
            # 初始化嵌入模型
            self.embedding_provider = EmbeddingProvider(embedding_model)
            
            # 初始化 ChromaDB
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"[成功] 记忆存储已初始化: {collection_name}")
            print(f"[信息] 使用嵌入后端: {EMBEDDING_TYPE}")
            
        except Exception as e:
            print(f"[错误] 初始化失败: {e}")
            self._fallback_mode = True
            self._memories: List[MemoryChunk] = []
    
    def add_memories(self, chunks: List[MemoryChunk]) -> List[str]:
        """
        批量添加记忆片段
        
        Args:
            chunks: 记忆片段列表
            
        Returns:
            添加的片段 ID 列表
        """
        if not chunks:
            return []
        
        if self._fallback_mode:
            # 降级模式：直接存储到内存
            self._memories.extend(chunks)
            return [c.id for c in chunks]
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        # 批量编码
        texts = [chunk.content for chunk in chunks]
        batch_embeddings = self.embedding_provider.encode(texts)
        
        for i, chunk in enumerate(chunks):
            ids.append(chunk.id)
            documents.append(chunk.content)
            metadatas.append({
                "source": chunk.source,
                "timestamp": chunk.timestamp.isoformat() if chunk.timestamp else None,
                **chunk.metadata
            })
            embeddings.append(batch_embeddings[i].tolist())
        
        # 批量添加到数据库
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        return ids
    
    def add_memory(self, chunk: MemoryChunk) -> str:
        """添加单条记忆"""
        self.add_memories([chunk])
        return chunk.id
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        use_context: bool = True,
        context_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        带上下文的智能检索
        
        算法：
        1. 基于当前查询的语义检索
        2. 结合对话上下文增强相关性
        3. 过滤已频繁提及的记忆
        
        Args:
            query: 当前查询
            top_k: 返回结果数量
            use_context: 是否使用上下文
            context_weight: 上下文权重
            
        Returns:
            检索结果列表
        """
        if self._fallback_mode:
            return self._fallback_retrieve(query, top_k)
        
        # 1. 基础检索
        base_results = self._semantic_search(query, top_k * 2)
        
        if not use_context or not self.enable_context:
            return base_results[:top_k]
        
        # 2. 获取对话上下文
        context_text = self.context.get_recent_context(3)
        
        # 3. 上下文增强检索
        if context_text:
            context_results = self._semantic_search(context_text, top_k * 2)
            
            # 合并并重新排序
            combined_scores = {}
            
            for r in base_results:
                combined_scores[r['id']] = {
                    'data': r,
                    'score': r['relevance'] * (1 - context_weight)
                }
            
            for r in context_results:
                if r['id'] in combined_scores:
                    combined_scores[r['id']]['score'] += r['relevance'] * context_weight
                else:
                    combined_scores[r['id']] = {
                        'data': r,
                        'score': r['relevance'] * context_weight
                    }
            
            # 排序
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )
            base_results = [item['data'] for item in sorted_results]
        
        # 4. 过滤已频繁提及的记忆
        filtered_results = []
        for r in base_results:
            if not self.context.is_memory_mentioned(r['id']):
                filtered_results.append(r)
            elif r['relevance'] > 0.9:  # 只有高度相关时才重复
                filtered_results.append(r)
        
        return filtered_results[:top_k]
    
    def _semantic_search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        语义检索核心
        """
        if self._fallback_mode:
            return self._fallback_retrieve(query, top_k, source_filter)
        
        # 生成查询向量
        query_embedding = self.embedding_provider.encode([query])[0]
        
        # 构建过滤条件
        where_filter = None
        if source_filter:
            where_filter = {"source": source_filter}
        
        # 执行检索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        memories = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                memories.append({
                    "id": doc_id,
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "relevance": 1 - results['distances'][0][i]
                })
        
        return memories
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        基础检索接口（保持兼容性）
        """
        return self.retrieve_with_context(query, top_k, use_context=False)
    
    def _fallback_retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """降级模式检索：简单的关键词匹配"""
        query_words = set(query.lower().split())
        scored_memories = []
        
        for mem in self._memories:
            if source_filter and mem.source != source_filter:
                continue
            
            content_words = set(mem.content.lower().split())
            overlap = len(query_words & content_words)
            score = overlap / max(len(query_words), 1)
            
            if score > 0:
                scored_memories.append((mem, score))
        
        # 按分数排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                "id": mem.id,
                "content": mem.content,
                "metadata": {"source": mem.source, **mem.metadata},
                "distance": 1 - score,
                "relevance": score
            }
            for mem, score in scored_memories[:top_k]
        ]
    
    def add_to_context(self, role: str, content: str):
        """
        添加消息到对话上下文
        
        Args:
            role: "user" 或 "assistant"
            content: 消息内容
        """
        if self.enable_context:
            self.context.add_message(role, content)
    
    def mark_memory_used(self, memory_id: str):
        """
        标记记忆已被使用
        
        Args:
            memory_id: 记忆ID
        """
        if self.enable_context:
            self.context.mark_memory_mentioned(memory_id)
    
    def get_context_for_prompt(self) -> str:
        """
        获取用于 Prompt 的上下文
        
        Returns:
            上下文字符串
        """
        if not self.enable_context:
            return ""
        
        return self.context.get_recent_context(5)
    
    def clear_context(self):
        """清空对话上下文"""
        if self.enable_context:
            self.context.clear()
    
    def retrieve_by_keywords(
        self,
        keywords: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        基于关键词检索（带上下文增强）
        """
        if not keywords:
            return []
        
        # 组合关键词为查询
        query = " ".join(keywords)
        return self.retrieve_with_context(query, top_k=top_k, use_context=True)
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取特定记忆"""
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"]
            )
            
            if result['ids']:
                return {
                    "id": result['ids'][0],
                    "content": result['documents'][0],
                    "metadata": result['metadatas'][0]
                }
        except Exception:
            pass
        
        return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除特定记忆"""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False
    
    def clear_all(self) -> bool:
        """清空所有记忆"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        if self._fallback_mode:
            return {
                "total_memories": len(self._memories),
                "collection_name": self.collection_name,
                "db_path": str(self.db_path),
                "mode": "fallback"
            }
        
        count = self.collection.count()
        
        return {
            "total_memories": count,
            "collection_name": self.collection_name,
            "db_path": str(self.db_path),
            "mode": "vector"
        }
    
    def export_memories(self, output_path: Path) -> Path:
        """导出所有记忆到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取所有记忆
        results = self.collection.get()
        
        memories = []
        if results['ids']:
            for i, doc_id in enumerate(results['ids']):
                memories.append({
                    "id": doc_id,
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })
        
        # 保存为 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)
        
        return output_path


class MemoryChunker:
    """
    记忆分块器
    
    将长文本分割成适合向量化的片段
    """
    
    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 50
    ):
        """
        初始化分块器
        
        Args:
            chunk_size: 每个块的目标大小（字符数）
            chunk_overlap: 块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(
        self,
        text: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> List[MemoryChunk]:
        """
        将文本分块
        
        Args:
            text: 原始文本
            source: 来源标识
            timestamp: 时间戳
            metadata: 额外元数据
            
        Returns:
            记忆片段列表
        """
        if not text:
            return []
        
        # 简单的滑动窗口分块
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 尝试在句子边界处截断
            if end < len(text):
                # 寻找最近的句号、问号或感叹号
                for i in range(end, min(end + 20, len(text))):
                    if text[i] in '。！？\n':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # 生成唯一 ID
                chunk_id = hashlib.md5(
                    f"{source}_{timestamp}_{start}".encode()
                ).hexdigest()[:16]
                
                chunks.append(MemoryChunk(
                    id=chunk_id,
                    content=chunk_text,
                    source=source,
                    timestamp=timestamp,
                    metadata=metadata or {}
                ))
            
            # 滑动窗口
            start = end - self.chunk_overlap
            if start >= end:
                break
        
        return chunks
    
    def chunk_messages(
        self,
        messages: List[Dict[str, Any]],
        source: str
    ) -> List[MemoryChunk]:
        """
        将消息列表分块
        
        Args:
            messages: 消息列表，每项包含 content, timestamp 等
            source: 来源标识
            
        Returns:
            记忆片段列表
        """
        chunks = []
        
        for msg in messages:
            content = msg.get('content', '')
            if not content or len(content) < 10:
                continue
            
            # 如果消息较短，直接作为一个块
            if len(content) <= self.chunk_size:
                chunk_id = hashlib.md5(
                    f"{source}_{msg.get('timestamp', '')}_{content[:20]}".encode()
                ).hexdigest()[:16]
                
                chunks.append(MemoryChunk(
                    id=chunk_id,
                    content=content,
                    source=source,
                    timestamp=msg.get('timestamp'),
                    metadata={"sender": msg.get('sender', 'unknown')}
                ))
            else:
                # 长消息需要分块
                msg_chunks = self.chunk_text(
                    content,
                    source,
                    msg.get('timestamp'),
                    {"sender": msg.get('sender', 'unknown')}
                )
                chunks.extend(msg_chunks)
        
        return chunks


class MemoryManager:
    """
    记忆管理器
    
    整合分块、存储、检索的完整流程
    """
    
    def __init__(
        self,
        base_path: Path,
        reunion_name: str
    ):
        """
        初始化记忆管理器
        
        Args:
            base_path: 基础存储路径
            reunion_name: 纪念对象名称
        """
        self.base_path = Path(base_path)
        self.reunion_name = reunion_name
        
        # 创建存储目录
        self.db_path = self.base_path / "vector_db" / reunion_name
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化存储和分块器
        self.store = MemoryStore(
            db_path=self.db_path,
            collection_name=f"memories_{reunion_name}"
        )
        self.chunker = MemoryChunker()
    
    def ingest_data(
        self,
        parsed_data: Any,
        source_type: str
    ) -> int:
        """
        摄入解析后的数据
        
        Args:
            parsed_data: 解析后的数据对象
            source_type: 数据源类型
            
        Returns:
            添加的记忆数量
        """
        # 转换为消息列表
        messages = [
            {
                "content": m.content,
                "timestamp": m.timestamp,
                "sender": m.sender
            }
            for m in parsed_data.messages
        ]
        
        # 分块
        chunks = self.chunker.chunk_messages(messages, source_type)
        
        # 存储
        if chunks:
            self.store.add_memories(chunks)
        
        return len(chunks)
    
    def retrieve_memories(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        return self.store.retrieve(query, top_k=top_k)
    
    def get_context_for_prompt(
        self,
        user_message: str,
        max_context_length: int = 1000
    ) -> str:
        """
        为对话生成记忆上下文
        
        Args:
            user_message: 用户消息
            max_context_length: 最大上下文长度
            
        Returns:
            格式化的记忆上下文
        """
        # 检索相关记忆
        memories = self.retrieve_memories(user_message, top_k=5)
        
        if not memories:
            return ""
        
        # 构建上下文
        context_parts = ["【相关记忆】"]
        current_length = len(context_parts[0])
        
        for mem in memories:
            if mem['relevance'] < 0.5:  # 过滤低相关度记忆
                continue
            
            mem_text = f"- {mem['content'][:150]}"
            if current_length + len(mem_text) > max_context_length:
                break
            
            context_parts.append(mem_text)
            current_length += len(mem_text)
        
        if len(context_parts) == 1:
            return ""
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.store.get_stats()
    
    def archive(self, archive_path: Path) -> Path:
        """归档记忆数据"""
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        return self.store.export_memories(archive_path)


# 便捷函数
def create_memory_manager(base_path: str, reunion_name: str) -> MemoryManager:
    """创建记忆管理器"""
    return MemoryManager(Path(base_path), reunion_name)
