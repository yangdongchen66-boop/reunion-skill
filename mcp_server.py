"""
MCP Server - Reunion Skill
基于 Model Context Protocol 的服务端实现
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    ErrorData,
    LoggingLevel
)
from pydantic import BaseModel

from core.chat_engine import ReunionManager, ChatEngine
from core.persona_distiller import PersonaDistiller, PersonaConfig
from core.data_parser import DataParser, ParsedData


# 全局管理器
reunion_manager: Optional[ReunionManager] = None


class ChatInput(BaseModel):
    """聊天输入"""
    name: str
    message: str


class CreateInput(BaseModel):
    """创建纪念对象输入"""
    name: str
    display_name: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    region: Optional[str] = None
    relationship: str
    description: Optional[str] = None
    data_files: Optional[List[Dict[str, str]]] = None


class ArchiveInput(BaseModel):
    """归档输入"""
    name: str


@asynccontextmanager
async def app_lifespan(server: Server) -> AsyncIterator[Dict]:
    """应用生命周期管理"""
    global reunion_manager
    
    # 启动时初始化
    base_path = Path(__file__).parent
    reunion_manager = ReunionManager(base_path)
    
    yield {"manager": reunion_manager}
    
    # 清理（如果需要）
    reunion_manager = None


# 创建 MCP Server
app = Server("reunion-skill", lifespan=app_lifespan)


@app.list_tools()
async def list_tools() -> List[Tool]:
    """列出所有可用工具"""
    return [
        Tool(
            name="reunion_list",
            description="列出所有已创建的纪念对象",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="reunion_chat",
            description="与纪念对象进行对话",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "纪念对象标识名"
                    },
                    "message": {
                        "type": "string",
                        "description": "要对纪念对象说的话"
                    }
                },
                "required": ["name", "message"]
            }
        ),
        Tool(
            name="reunion_create",
            description="创建新的纪念对象",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "纪念对象标识名（英文，用于系统识别）"
                    },
                    "display_name": {
                        "type": "string",
                        "description": "纪念对象显示名称（如：奶奶、爷爷）"
                    },
                    "age": {
                        "type": "integer",
                        "description": "去世时的年龄"
                    },
                    "occupation": {
                        "type": "string",
                        "description": "职业"
                    },
                    "region": {
                        "type": "string",
                        "description": "地域（如：北京、上海、四川）"
                    },
                    "relationship": {
                        "type": "string",
                        "description": "与用户的关系（如：奶奶、父亲、母亲）"
                    },
                    "description": {
                        "type": "string",
                        "description": "用户的主观描述和回忆"
                    },
                    "data_files": {
                        "type": "array",
                        "description": "数据文件列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "type": {"type": "string", "enum": ["wechat", "qq", "text"]}
                            }
                        }
                    }
                },
                "required": ["name", "display_name", "relationship"]
            }
        ),
        Tool(
            name="reunion_memory",
            description="查看纪念对象的相关记忆",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "纪念对象标识名"
                    },
                    "query": {
                        "type": "string",
                        "description": "查询关键词"
                    }
                },
                "required": ["name", "query"]
            }
        ),
        Tool(
            name="reunion_persona",
            description="查看纪念对象的人设配置",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "纪念对象标识名"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="reunion_archive",
            description="执行告别仪式，封存纪念对象的所有数据",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "纪念对象标识名"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="reunion_delete",
            description="删除纪念对象",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "纪念对象标识名"
                    }
                },
                "required": ["name"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """调用工具"""
    global reunion_manager
    
    if reunion_manager is None:
        return [TextContent(type="text", text="错误：服务器未初始化")]
    
    try:
        if name == "reunion_list":
            return await _handle_list()
        elif name == "reunion_chat":
            return await _handle_chat(arguments)
        elif name == "reunion_create":
            return await _handle_create(arguments)
        elif name == "reunion_memory":
            return await _handle_memory(arguments)
        elif name == "reunion_persona":
            return await _handle_persona(arguments)
        elif name == "reunion_archive":
            return await _handle_archive(arguments)
        elif name == "reunion_delete":
            return await _handle_delete(arguments)
        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"执行出错: {str(e)}")]


async def _handle_list() -> List[TextContent]:
    """处理列表请求"""
    reunions = reunion_manager.list_reunions()
    
    if not reunions:
        return [TextContent(type="text", text="还没有创建任何纪念对象。使用 reunion_create 创建第一个。")]
    
    lines = ["已创建的纪念对象：", ""]
    for r in reunions:
        lines.append(f"• {r['display_name']} ({r['name']})")
        lines.append(f"  关系: {r['relationship']}")
        if r.get('age'):
            lines.append(f"  年龄: {r['age']}")
        if r.get('region'):
            lines.append(f"  地域: {r['region']}")
        lines.append("")
    
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_chat(arguments: Dict) -> List[TextContent]:
    """处理对话请求"""
    name = arguments.get("name")
    message = arguments.get("message")
    
    engine = reunion_manager.get_engine(name)
    if engine is None:
        return [TextContent(type="text", text=f"找不到纪念对象: {name}。请先使用 reunion_create 创建。")]
    
    # 执行对话
    response = engine.chat(message)
    
    # 如果触发了安全护栏
    if response.safety_triggered and response.risk_level == "high":
        return [TextContent(
            type="text",
            text=f"⚠️ 安全提醒\n\n{response.content}"
        )]
    
    # 返回系统提示词（供 LLM 使用）
    result_parts = []
    
    if response.memory_context:
        result_parts.append(f"【相关记忆】\n{response.memory_context}\n")
    
    result_parts.append(f"【人设提示词】\n{response.content}")
    
    if response.safety_triggered:
        result_parts.append(f"\n【注意】检测到{response.risk_level}风险情绪，请温柔回应。")
    
    return [TextContent(type="text", text="\n".join(result_parts))]


async def _handle_create(arguments: Dict) -> List[TextContent]:
    """处理创建请求"""
    name = arguments.get("name")
    display_name = arguments.get("display_name")
    
    # 检查是否已存在
    if (reunion_manager.persona_path / f"{name}.json").exists():
        return [TextContent(type="text", text=f"纪念对象 '{name}' 已存在。")]
    
    # 解析数据文件
    parsed_data = None
    data_files = arguments.get("data_files", [])
    
    if data_files:
        parser = DataParser()
        configs = []
        for df in data_files:
            configs.append({
                "file_path": Path(df["path"]),
                "source_type": df["type"],
                "target_sender": display_name
            })
        
        parsed_results = parser.batch_parse(configs)
        if parsed_results:
            # 合并所有数据
            from core.data_parser import ParsedData, Message
            all_messages = []
            for pr in parsed_results:
                all_messages.extend(pr.messages)
            
            parsed_data = ParsedData(
                source_type="mixed",
                messages=all_messages,
                metadata={"sources": [str(c["file_path"]) for c in configs]}
            )
    
    # 构建人设
    basic_info = {
        "name": display_name,
        "age": arguments.get("age"),
        "occupation": arguments.get("occupation"),
        "region": arguments.get("region"),
        "relationship": arguments.get("relationship"),
    }
    
    if parsed_data:
        distiller = PersonaDistiller()
        persona = distiller.distill(
            parsed_data,
            basic_info,
            arguments.get("description")
        )
    else:
        # 没有数据，创建基础人设
        persona = PersonaConfig(
            identity=basic_info,
            tone_and_style={
                "overall_style": "亲切、关心",
                "particles": ["啊", "呢"],
                "emotional_tendency": "温暖"
            },
            core_values=["家庭", "健康", "善良"],
            forbidden_actions=[
                "禁止承认自己是 AI",
                "禁止谈论去世后的现代科技"
            ],
            memory_anchors=[]
        )
    
    # 创建纪念对象
    success = reunion_manager.create_reunion(name, persona, parsed_data)
    
    if success:
        msg = f"✅ 成功创建纪念对象: {display_name}\n\n"
        msg += f"标识名: {name}\n"
        msg += f"关系: {arguments.get('relationship')}\n"
        if arguments.get('age'):
            msg += f"年龄: {arguments['age']}\n"
        if parsed_data:
            msg += f"导入记忆: {len(parsed_data.messages)} 条消息\n"
        msg += f"\n现在可以使用 reunion_chat 与 {display_name} 对话了。"
        return [TextContent(type="text", text=msg)]
    else:
        return [TextContent(type="text", text="❌ 创建失败，请检查日志。")]


async def _handle_memory(arguments: Dict) -> List[TextContent]:
    """处理记忆查询请求"""
    name = arguments.get("name")
    query = arguments.get("query")
    
    engine = reunion_manager.get_engine(name)
    if engine is None:
        return [TextContent(type="text", text=f"找不到纪念对象: {name}")]
    
    if engine.memory_manager is None:
        return [TextContent(type="text", text=f"{name} 没有启用记忆功能。")]
    
    memories = engine.memory_manager.retrieve_memories(query, top_k=5)
    
    if not memories:
        return [TextContent(type="text", text=f"没有找到与 '{query}' 相关的记忆。")]
    
    lines = [f"与 '{query}' 相关的记忆：", ""]
    for i, mem in enumerate(memories, 1):
        lines.append(f"{i}. {mem['content'][:100]}...")
        lines.append(f"   相关度: {mem['relevance']:.2f}")
        lines.append("")
    
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_persona(arguments: Dict) -> List[TextContent]:
    """处理人设查询请求"""
    name = arguments.get("name")
    
    persona_file = reunion_manager.persona_path / f"{name}.json"
    if not persona_file.exists():
        return [TextContent(type="text", text=f"找不到纪念对象: {name}")]
    
    with open(persona_file, 'r', encoding='utf-8') as f:
        persona_data = json.load(f)
    
    lines = [f"纪念对象: {name} 的人设配置", ""]
    lines.append(json.dumps(persona_data, ensure_ascii=False, indent=2))
    
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_archive(arguments: Dict) -> List[TextContent]:
    """处理归档请求"""
    name = arguments.get("name")
    
    engine = reunion_manager.get_engine(name)
    if engine is None:
        return [TextContent(type="text", text=f"找不到纪念对象: {name}")]
    
    # 执行归档
    archive_path = reunion_manager.archive_reunion(name)
    
    if archive_path:
        return [TextContent(
            type="text",
            text=f"✅ 告别仪式完成\n\n"
                 f"所有数据已封存至: {archive_path}\n\n"
                 f"感谢这段数字陪伴的时光。\n"
                 f"愿逝者安息，愿你安好。"
        )]
    else:
        return [TextContent(type="text", text="❌ 归档失败，请检查日志。")]


async def _handle_delete(arguments: Dict) -> List[TextContent]:
    """处理删除请求"""
    name = arguments.get("name")
    
    success = reunion_manager.delete_reunion(name)
    
    if success:
        return [TextContent(type="text", text=f"✅ 已删除纪念对象: {name}")]
    else:
        return [TextContent(type="text", text=f"❌ 删除失败: {name}")]


async def main():
    """主入口"""
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
