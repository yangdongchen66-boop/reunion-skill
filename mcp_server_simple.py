#!/usr/bin/env python3
"""
MCP Server - Reunion Skill (简化版)
基于 Model Context Protocol 的服务端实现
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.chat_engine import ReunionManager, ChatEngine
from core.persona_distiller import PersonaDistiller, PersonaConfig
from core.data_parser import DataParser, ParsedData


# 全局管理器
BASE_PATH = Path(__file__).parent
reunion_manager = ReunionManager(BASE_PATH)


def handle_list(arguments: Dict = None) -> Dict:
    """处理列表请求"""
    reunions = reunion_manager.list_reunions()
    
    if not reunions:
        return {"content": "还没有创建任何纪念对象。使用 reunion_create 创建第一个。"}
    
    lines = ["已创建的纪念对象：", ""]
    for r in reunions:
        lines.append(f"• {r['display_name']} ({r['name']})")
        lines.append(f"  关系: {r['relationship']}")
        if r.get('age'):
            lines.append(f"  年龄: {r['age']}")
        if r.get('region'):
            lines.append(f"  地域: {r['region']}")
        lines.append("")
    
    return {"content": "\n".join(lines)}


def handle_chat(arguments: Dict) -> Dict:
    """处理对话请求"""
    name = arguments.get("name")
    message = arguments.get("message")
    
    engine = reunion_manager.get_engine(name)
    if engine is None:
        return {"content": f"找不到纪念对象: {name}。请先使用 reunion_create 创建。", "isError": True}
    
    # 执行对话
    response = engine.chat(message)
    
    # 如果触发了安全护栏
    if response.safety_triggered and response.risk_level == "high":
        return {"content": f"⚠️ 安全提醒\n\n{response.content}"}
    
    # 返回系统提示词（供 LLM 使用）
    result_parts = []
    
    if response.memory_context:
        result_parts.append(f"【相关记忆】\n{response.memory_context}\n")
    
    result_parts.append(f"【人设提示词】\n{response.content}")
    
    if response.safety_triggered:
        result_parts.append(f"\n【注意】检测到{response.risk_level}风险情绪，请温柔回应。")
    
    return {"content": "\n".join(result_parts)}


def handle_create(arguments: Dict) -> Dict:
    """处理创建请求"""
    name = arguments.get("name")
    display_name = arguments.get("display_name")
    
    # 检查是否已存在
    if (reunion_manager.persona_path / f"{name}.json").exists():
        return {"content": f"纪念对象 '{name}' 已存在。", "isError": True}
    
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
        return {"content": msg}
    else:
        return {"content": "❌ 创建失败，请检查日志。", "isError": True}


def handle_persona(arguments: Dict) -> Dict:
    """处理人设查询请求"""
    name = arguments.get("name")
    
    persona_file = reunion_manager.persona_path / f"{name}.json"
    if not persona_file.exists():
        return {"content": f"找不到纪念对象: {name}", "isError": True}
    
    with open(persona_file, 'r', encoding='utf-8') as f:
        persona_data = json.load(f)
    
    lines = [f"纪念对象: {name} 的人设配置", ""]
    lines.append(json.dumps(persona_data, ensure_ascii=False, indent=2))
    
    return {"content": "\n".join(lines)}


def handle_delete(arguments: Dict) -> Dict:
    """处理删除请求"""
    name = arguments.get("name")
    
    success = reunion_manager.delete_reunion(name)
    
    if success:
        return {"content": f"✅ 已删除纪念对象: {name}"}
    else:
        return {"content": f"❌ 删除失败: {name}", "isError": True}


def handle_archive(arguments: Dict) -> Dict:
    """处理归档请求"""
    name = arguments.get("name")
    
    engine = reunion_manager.get_engine(name)
    if engine is None:
        return {"content": f"找不到纪念对象: {name}", "isError": True}
    
    # 执行归档
    archive_path = reunion_manager.archive_reunion(name)
    
    if archive_path:
        return {"content": f"✅ 告别仪式完成\n\n所有数据已封存至: {archive_path}\n\n感谢这段数字陪伴的时光。\n愿逝者安息，愿你安好。"}
    else:
        return {"content": "❌ 归档失败，请检查日志。", "isError": True}


# MCP 工具定义
TOOLS = [
    {
        "name": "reunion_list",
        "description": "列出所有已创建的纪念对象",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "reunion_chat",
        "description": "与纪念对象进行对话",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "纪念对象标识名"},
                "message": {"type": "string", "description": "要对纪念对象说的话"}
            },
            "required": ["name", "message"]
        }
    },
    {
        "name": "reunion_create",
        "description": "创建新的纪念对象",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "纪念对象标识名（英文）"},
                "display_name": {"type": "string", "description": "显示名称（如：奶奶）"},
                "age": {"type": "integer", "description": "去世时的年龄"},
                "occupation": {"type": "string", "description": "职业"},
                "region": {"type": "string", "description": "地域"},
                "relationship": {"type": "string", "description": "与用户的关系"},
                "description": {"type": "string", "description": "主观描述"},
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
    },
    {
        "name": "reunion_persona",
        "description": "查看纪念对象的人设配置",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "纪念对象标识名"}},
            "required": ["name"]
        }
    },
    {
        "name": "reunion_archive",
        "description": "执行告别仪式，封存纪念对象的所有数据",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "纪念对象标识名"}},
            "required": ["name"]
        }
    },
    {
        "name": "reunion_delete",
        "description": "删除纪念对象",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "纪念对象标识名"}},
            "required": ["name"]
        }
    },
]


def handle_request(request: Dict) -> Dict:
    """处理 MCP 请求"""
    method = request.get("method")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "reunion-skill", "version": "0.1.0"},
                "capabilities": {"tools": {}}
            }
        }
    
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"tools": TOOLS}
        }
    
    elif method == "tools/call":
        tool_name = request.get("params", {}).get("name")
        arguments = request.get("params", {}).get("arguments", {})
        
        handlers = {
            "reunion_list": handle_list,
            "reunion_chat": handle_chat,
            "reunion_create": handle_create,
            "reunion_persona": handle_persona,
            "reunion_archive": handle_archive,
            "reunion_delete": handle_delete,
        }
        
        handler = handlers.get(tool_name)
        if handler:
            result = handler(arguments)
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [{"type": "text", "text": result.get("content", "")}],
                    "isError": result.get("isError", False)
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
            }
    
    return {"jsonrpc": "2.0", "id": request.get("id"), "error": {"code": -32600, "message": "Invalid request"}}


async def main():
    """主入口 - 简化的 stdio 服务器"""
    import sys
    
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            
            request = json.loads(line.strip())
            response = handle_request(request)
            
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(e)}
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
