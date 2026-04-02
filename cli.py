#!/usr/bin/env python3
"""
Reunion Skill CLI
命令行交互工具
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from core.chat_engine import ReunionManager
from core.persona_distiller import PersonaDistiller, PersonaConfig
from core.data_parser import DataParser


app = typer.Typer(help="Reunion Skill - 重逢")
console = Console()

# 初始化管理器
BASE_PATH = Path(__file__).parent
manager = ReunionManager(BASE_PATH)


@app.command()
def list():
    """列出所有纪念对象"""
    reunions = manager.list_reunions()
    
    if not reunions:
        console.print("[yellow]还没有创建任何纪念对象。[/yellow]")
        return
    
    table = Table(title="纪念对象列表")
    table.add_column("名称", style="cyan")
    table.add_column("关系", style="magenta")
    table.add_column("年龄", style="green")
    table.add_column("地域", style="blue")
    
    for r in reunions:
        table.add_row(
            r['display_name'],
            r['relationship'],
            str(r.get('age', '-')),
            r.get('region', '-')
        )
    
    console.print(table)


@app.command()
def create(
    name: Optional[str] = typer.Argument(None, help="标识名（英文）"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="交互模式")
):
    """创建新的纪念对象"""
    
    if interactive or not name:
        console.print(Panel.fit("创建新的纪念对象", style="cyan"))
        
        name = Prompt.ask("请输入标识名（英文，用于系统识别）", default=name or "")
        display_name = Prompt.ask("请输入显示名称（如：奶奶、爷爷）")
        relationship = Prompt.ask("与你的关系", choices=["奶奶", "爷爷", "父亲", "母亲", "其他"], default="奶奶")
        
        age_str = Prompt.ask("去世时的年龄（可选）", default="")
        age = int(age_str) if age_str.isdigit() else None
        
        occupation = Prompt.ask("职业（可选）", default="")
        region = Prompt.ask("地域（可选）", default="")
        
        console.print("\n[dim]请描述一下 ta 的性格特点、口头禅、价值观等：[/dim]")
        description = Prompt.ask("描述")
        
        # 询问数据文件
        has_data = Confirm.ask("是否有聊天记录或日记文件需要导入？")
        data_files = []
        
        if has_data:
            while True:
                file_path = Prompt.ask("文件路径（留空结束）", default="")
                if not file_path:
                    break
                file_type = Prompt.ask("文件类型", choices=["wechat", "qq", "text"], default="wechat")
                data_files.append({"path": file_path, "type": file_type})
    else:
        # 非交互模式，使用默认值
        display_name = name
        relationship = "未知"
        age = None
        occupation = None
        region = None
        description = None
        data_files = []
    
    # 解析数据
    parsed_data = None
    if data_files:
        with console.status("[bold green]正在解析数据..."):
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
                from core.data_parser import ParsedData
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
        "age": age,
        "occupation": occupation,
        "region": region,
        "relationship": relationship,
    }
    
    with console.status("[bold green]正在生成人设..."):
        if parsed_data:
            distiller = PersonaDistiller()
            persona = distiller.distill(parsed_data, basic_info, description)
        else:
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
    
    # 创建
    with console.status("[bold green]正在创建..."):
        success = manager.create_reunion(name, persona, parsed_data)
    
    if success:
        console.print(f"\n[green]✓[/green] 成功创建纪念对象: [bold]{display_name}[/bold]")
        if parsed_data:
            console.print(f"  导入记忆: {len(parsed_data.messages)} 条消息")
        console.print(f"\n使用 [cyan]reunion chat {name}[/cyan] 开始对话")
    else:
        console.print("[red]✗[/red] 创建失败")


@app.command()
def chat(
    name: str = typer.Argument(..., help="纪念对象名称"),
    message: Optional[str] = typer.Argument(None, help="消息内容")
):
    """与纪念对象对话"""
    engine = manager.get_engine(name)
    
    if engine is None:
        console.print(f"[red]找不到纪念对象: {name}[/red]")
        console.print(f"使用 [cyan]reunion create {name}[/cyan] 创建")
        return
    
    if message:
        # 单条消息模式
        response = engine.chat(message)
        
        if response.safety_triggered and response.risk_level == "high":
            console.print(Panel(response.content, title="⚠️ 安全提醒", style="red"))
        else:
            # 显示提示词（实际使用时这里应该调用 LLM）
            console.print(Panel(response.content, title="系统提示词", style="dim"))
            console.print("\n[yellow]提示：以上是给 LLM 的系统提示词，实际回复需要由 LLM 生成。[/yellow]")
    else:
        # 交互模式
        console.print(Panel.fit(f"与 {name} 的对话", style="cyan"))
        console.print("[dim]输入 /quit 退出对话[/dim]\n")
        
        while True:
            user_input = Prompt.ask("你")
            
            if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                break
            
            response = engine.chat(user_input)
            
            if response.safety_triggered and response.risk_level == "high":
                console.print(Panel(response.content, title="⚠️ 安全提醒", style="red"))
            else:
                # 显示提示词
                console.print(Panel(response.content[:500] + "...", title="系统提示词（节选）", style="dim"))


@app.command()
def memory(
    name: str = typer.Argument(..., help="纪念对象名称"),
    query: str = typer.Argument(..., help="查询关键词")
):
    """查询记忆"""
    engine = manager.get_engine(name)
    
    if engine is None:
        console.print(f"[red]找不到纪念对象: {name}[/red]")
        return
    
    if engine.memory_manager is None:
        console.print(f"[yellow]{name} 没有启用记忆功能[/yellow]")
        return
    
    memories = engine.memory_manager.retrieve_memories(query, top_k=5)
    
    if not memories:
        console.print(f"[yellow]没有找到与 '{query}' 相关的记忆[/yellow]")
        return
    
    console.print(f"\n[bold]与 '{query}' 相关的记忆：[/bold]\n")
    
    for i, mem in enumerate(memories, 1):
        panel = Panel(
            mem['content'][:150] + "..." if len(mem['content']) > 150 else mem['content'],
            title=f"{i}. 相关度: {mem['relevance']:.2f}",
            style="blue"
        )
        console.print(panel)


@app.command()
def persona(name: str = typer.Argument(..., help="纪念对象名称")):
    """查看人设"""
    import json
    
    persona_file = manager.persona_path / f"{name}.json"
    if not persona_file.exists():
        console.print(f"[red]找不到纪念对象: {name}[/red]")
        return
    
    with open(persona_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    console.print(json.dumps(data, ensure_ascii=False, indent=2))


@app.command()
def archive(name: str = typer.Argument(..., help="纪念对象名称")):
    """执行告别仪式，封存数据"""
    if not Confirm.ask(f"确定要封存 {name} 的所有数据吗？这是一个不可逆的操作。"):
        console.print("[yellow]已取消[/yellow]")
        return
    
    with console.status("[bold green]正在执行告别仪式..."):
        archive_path = manager.archive_reunion(name)
    
    if archive_path:
        console.print(f"\n[green]✓[/green] 告别仪式完成")
        console.print(f"  数据已封存至: {archive_path}")
        console.print("\n[dim]感谢这段数字陪伴的时光。[/dim]")
        console.print("[dim]愿逝者安息，愿你安好。[/dim]")
    else:
        console.print("[red]✗[/red] 归档失败")


@app.command()
def delete(name: str = typer.Argument(..., help="纪念对象名称")):
    """删除纪念对象"""
    if not Confirm.ask(f"确定要删除 {name} 吗？此操作不可恢复。"):
        console.print("[yellow]已取消[/yellow]")
        return
    
    success = manager.delete_reunion(name)
    
    if success:
        console.print(f"[green]✓[/green] 已删除 {name}")
    else:
        console.print(f"[red]✗[/red] 删除失败")


@app.callback()
def main():
    """
    Reunion Skill - 重逢
    
    用 AI 的方式，让逝去的亲人以另一种形式继续陪伴。
    """
    pass


if __name__ == "__main__":
    app()
