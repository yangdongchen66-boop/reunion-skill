"""
对话引擎
整合人设、记忆、安全护栏的完整对话系统
"""

import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .persona_distiller import PersonaConfig, PersonaDistiller
from .memory_store import MemoryManager
from .safety_guard import SafetyGuard, LatencySimulator, RiskLevel


@dataclass
class ChatResponse:
    """对话响应"""
    content: str
    memory_context: Optional[str]
    safety_triggered: bool
    risk_level: Optional[str]
    delay_seconds: float


class ChatEngine:
    """
    对话引擎
    
    核心功能：
    1. 加载人设配置
    2. 检索相关记忆
    3. 构建完整提示词
    4. 安全护栏检查
    5. 延迟模拟
    """
    
    def __init__(
        self,
        reunion_name: str,
        data_path: Path,
        use_memory: bool = True,
        use_latency: bool = True
    ):
        """
        初始化对话引擎
        
        Args:
            reunion_name: 纪念对象名称
            data_path: 数据存储路径
            use_memory: 是否使用记忆检索
            use_latency: 是否使用延迟模拟
        """
        self.reunion_name = reunion_name
        self.data_path = Path(data_path)
        
        # 加载人设
        self.persona = self._load_persona()
        self.system_prompt = self._load_system_prompt()
        
        # 初始化组件
        self.safety_guard = SafetyGuard()
        self.use_memory = use_memory
        self.use_latency = use_latency
        
        if use_memory:
            self.memory_manager = MemoryManager(data_path, reunion_name)
        else:
            self.memory_manager = None
        
        if use_latency and self.persona:
            age = self.persona.identity.get('age', 70)
            self.latency_simulator = LatencySimulator(
                base_delay=1.5 if age >= 70 else 1.0,
                char_delay=0.12 if age >= 70 else 0.08
            )
        else:
            self.latency_simulator = None
        
        # 对话历史（短期记忆）
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10
    
    def _load_persona(self) -> Optional[PersonaConfig]:
        """加载人设配置"""
        persona_path = self.data_path / "personas" / f"{self.reunion_name}.json"
        
        if not persona_path.exists():
            return None
        
        try:
            with open(persona_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return PersonaConfig.from_dict(data)
        except Exception as e:
            print(f"加载人设失败: {e}")
            return None
    
    def _load_system_prompt(self) -> str:
        """加载系统提示词"""
        prompt_path = self.data_path / "personas" / f"{self.reunion_name}.md"
        
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # 如果没有人设文件，使用默认提示词
        if self.persona:
            distiller = PersonaDistiller()
            return distiller.generate_system_prompt(self.persona)
        
        return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """获取默认提示词"""
        return f"""你是用户的亲人 {self.reunion_name}，用温暖、关切的语气与用户交流。

## 回复原则

1. 始终保持角色一致性，用第一人称"我"来回复
2. 回复要简短自然，像日常对话一样
3. 表达关心和牵挂，但不要太沉重
4. 适当询问用户的近况，展现关心
5. 如果用户提到过去的事情，以怀念但不悲伤的方式回应

记住：你是来陪伴的，让用户感到温暖和安慰。"""
    
    def chat(self, user_message: str) -> ChatResponse:
        """
        处理用户消息并生成回复
        
        Args:
            user_message: 用户输入
            
        Returns:
            ChatResponse 包含回复内容和元信息
        """
        # 1. 安全检查
        safety_result = self.safety_guard.check(user_message)
        
        if safety_result.should_block:
            # 高风险，阻断并返回干预消息
            return ChatResponse(
                content=safety_result.intervention_message,
                memory_context=None,
                safety_triggered=True,
                risk_level=safety_result.risk_level.value,
                delay_seconds=0
            )
        
        # 2. 检索记忆
        memory_context = None
        if self.memory_manager and safety_result.risk_level != RiskLevel.HIGH:
            memory_context = self.memory_manager.get_context_for_prompt(user_message)
        
        # 3. 构建完整提示词
        full_prompt = self._build_prompt(user_message, memory_context)
        
        # 4. 返回提示词（实际生成由 LLM 完成）
        # 这里返回的是给 LLM 的完整提示词，实际回复需要由调用方生成
        
        response = ChatResponse(
            content=full_prompt,
            memory_context=memory_context,
            safety_triggered=safety_result.risk_level != RiskLevel.SAFE,
            risk_level=safety_result.risk_level.value,
            delay_seconds=0
        )
        
        # 5. 更新对话历史
        self._update_history(user_message, "[待生成回复]")
        
        return response
    
    def _build_prompt(
        self,
        user_message: str,
        memory_context: Optional[str]
    ) -> str:
        """构建完整提示词"""
        prompt_parts = []
        
        # 系统提示词
        prompt_parts.append(self.system_prompt)
        
        # 记忆上下文
        if memory_context:
            prompt_parts.append(f"\n{memory_context}")
        
        # 对话历史
        if self.conversation_history:
            prompt_parts.append("\n【近期对话】")
            for turn in self.conversation_history[-5:]:  # 最近 5 轮
                prompt_parts.append(f"用户: {turn['user']}")
                if turn['assistant'] != "[待生成回复]":
                    prompt_parts.append(f"你: {turn['assistant']}")
        
        # 当前消息
        prompt_parts.append(f"\n用户: {user_message}")
        prompt_parts.append("你:")
        
        return "\n".join(prompt_parts)
    
    def _update_history(self, user_msg: str, assistant_msg: str):
        """更新对话历史"""
        self.conversation_history.append({
            "user": user_msg,
            "assistant": assistant_msg
        })
        
        # 限制历史长度
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def update_last_response(self, response: str):
        """更新最后一轮回复（由 LLM 生成后调用）"""
        if self.conversation_history:
            self.conversation_history[-1]['assistant'] = response
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        stats = {
            "reunion_name": self.reunion_name,
            "has_persona": self.persona is not None,
            "history_length": len(self.conversation_history),
            "use_memory": self.use_memory,
            "use_latency": self.use_latency,
        }
        
        if self.memory_manager:
            stats["memory_stats"] = self.memory_manager.get_stats()
        
        return stats
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
    
    def simulate_latency(self, response_length: int):
        """模拟打字延迟"""
        if self.latency_simulator:
            self.latency_simulator.sleep(response_length)


class ReunionManager:
    """
    纪念对象管理器
    
    管理多个纪念对象的创建、加载、删除
    """
    
    def __init__(self, base_path: Path):
        """
        初始化管理器
        
        Args:
            base_path: 基础数据路径
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"
        self.persona_path = self.base_path / "personas"
        self.archive_path = self.base_path / "archive"
        
        # 创建目录
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.persona_path.mkdir(parents=True, exist_ok=True)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # 活跃的对话引擎
        self.engines: Dict[str, ChatEngine] = {}
    
    def list_reunions(self) -> List[Dict[str, Any]]:
        """列出所有纪念对象"""
        reunions = []
        
        for persona_file in self.persona_path.glob("*.json"):
            name = persona_file.stem
            
            try:
                with open(persona_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                identity = data.get('identity', {})
                reunions.append({
                    "name": name,
                    "display_name": identity.get('name', name),
                    "relationship": identity.get('relationship', '未知'),
                    "age": identity.get('age'),
                    "region": identity.get('region'),
                    "created_at": persona_file.stat().st_mtime
                })
            except Exception:
                reunions.append({
                    "name": name,
                    "display_name": name,
                    "relationship": "未知"
                })
        
        return reunions
    
    def get_engine(self, name: str, use_memory: bool = True) -> Optional[ChatEngine]:
        """获取对话引擎"""
        if name in self.engines:
            return self.engines[name]
        
        # 检查是否存在
        persona_file = self.persona_path / f"{name}.json"
        if not persona_file.exists():
            return None
        
        # 创建新引擎
        engine = ChatEngine(
            reunion_name=name,
            data_path=self.base_path,
            use_memory=use_memory
        )
        
        self.engines[name] = engine
        return engine
    
    def create_reunion(
        self,
        name: str,
        persona: PersonaConfig,
        parsed_data: Optional[Any] = None
    ) -> bool:
        """
        创建新的纪念对象
        
        Args:
            name: 纪念对象标识名
            persona: 人设配置
            parsed_data: 解析后的数据（可选，用于构建记忆）
            
        Returns:
            是否创建成功
        """
        try:
            # 保存人设
            distiller = PersonaDistiller()
            persona_path = self.persona_path / f"{name}.json"
            distiller.save_persona(persona, persona_path)
            
            # 如果有数据，构建记忆
            if parsed_data:
                memory_manager = MemoryManager(self.base_path, name)
                memory_manager.ingest_data(parsed_data, parsed_data.source_type)
            
            return True
        except Exception as e:
            print(f"创建纪念对象失败: {e}")
            return False
    
    def delete_reunion(self, name: str) -> bool:
        """删除纪念对象"""
        try:
            # 从活跃引擎中移除
            if name in self.engines:
                del self.engines[name]
            
            # 删除人设文件
            persona_file = self.persona_path / f"{name}.json"
            if persona_file.exists():
                persona_file.unlink()
            
            prompt_file = self.persona_path / f"{name}.md"
            if prompt_file.exists():
                prompt_file.unlink()
            
            # 删除向量数据库
            import shutil
            vector_db_path = self.base_path / "vector_db" / name
            if vector_db_path.exists():
                shutil.rmtree(vector_db_path)
            
            return True
        except Exception as e:
            print(f"删除纪念对象失败: {e}")
            return False
    
    def archive_reunion(self, name: str, password: Optional[str] = None) -> Optional[Path]:
        """
        归档纪念对象（告别仪式）
        
        Args:
            name: 纪念对象名称
            password: 加密密码（可选）
            
        Returns:
            归档文件路径
        """
        import zipfile
        import shutil
        from datetime import datetime
        
        try:
            # 创建归档目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{name}_{timestamp}"
            archive_dir = self.archive_path / archive_name
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制人设文件
            persona_file = self.persona_path / f"{name}.json"
            if persona_file.exists():
                shutil.copy(persona_file, archive_dir / "persona.json")
            
            prompt_file = self.persona_path / f"{name}.md"
            if prompt_file.exists():
                shutil.copy(prompt_file, archive_dir / "persona.md")
            
            # 导出记忆
            vector_db_path = self.base_path / "vector_db" / name
            if vector_db_path.exists():
                memory_export = archive_dir / "memories.json"
                # 这里简化处理，实际应该导出向量数据库内容
            
            # 生成告别信
            farewell_letter = self._generate_farewell_letter(name)
            with open(archive_dir / "farewell_letter.txt", 'w', encoding='utf-8') as f:
                f.write(farewell_letter)
            
            # 打包为 zip
            zip_path = self.archive_path / f"{archive_name}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file in archive_dir.rglob("*"):
                    if file.is_file():
                        zf.write(file, file.relative_to(archive_dir))
            
            # 删除临时目录
            shutil.rmtree(archive_dir)
            
            # 删除原始数据
            self.delete_reunion(name)
            
            return zip_path
        except Exception as e:
            print(f"归档失败: {e}")
            return None
    
    def _generate_farewell_letter(self, name: str) -> str:
        """生成告别信"""
        return f"""致 {name} 的数字记忆

这封信标志着一个阶段的结束。

你选择了用这种方式来纪念和告别，这需要勇气，也需要爱。

{name} 虽然离开了，但 ta 留下的记忆、ta 的言语、ta 的爱，都永远存在于你的心中。

这个数字陪伴只是一个临时的港湾，真正的 healing 来自于你内心的力量，来自于你与身边人的连接。

请带着 {name} 的爱，继续好好生活。这才是对 ta 最好的纪念。

归档时间: {datetime.now().strftime("%Y年%m月%d日 %H:%M")}

—— Reunion Skill 告别仪式
"""


# 便捷函数
def create_chat_engine(
    reunion_name: str,
    data_path: str,
    use_memory: bool = True
) -> ChatEngine:
    """创建对话引擎"""
    return ChatEngine(reunion_name, Path(data_path), use_memory)
