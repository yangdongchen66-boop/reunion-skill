"""
数据解析器模块
负责解析微信、QQ、文本等多种数据源
"""

import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class Message:
    """单条消息数据结构"""
    timestamp: Optional[datetime]
    sender: str
    content: str
    message_type: str = "text"  # text, image, voice, system
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "sender": self.sender,
            "content": self.content,
            "message_type": self.message_type
        }


@dataclass
class ParsedData:
    """解析后的数据结构"""
    source_type: str  # wechat, qq, text, diary
    messages: List[Message]
    metadata: Dict
    
    def get_sender_messages(self, sender_name: str) -> List[Message]:
        """获取特定发送者的消息"""
        return [m for m in self.messages if m.sender == sender_name]
    
    def to_json(self) -> str:
        """导出为 JSON"""
        return json.dumps({
            "source_type": self.source_type,
            "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages]
        }, ensure_ascii=False, indent=2)


class WeChatParser:
    """微信聊天记录解析器
    
    支持格式：
    1. WeChatMsg 导出格式
    2. 留痕导出格式
    3. PyWxDump 导出格式
    4. 普通 txt 导出格式
    """
    
    # 系统消息过滤模式
    SYSTEM_PATTERNS = [
        r"撤回了一条消息",
        r"拍了拍",
        r"邀请.*加入了群聊",
        r"移出了群聊",
        r"修改群名为",
        r"开启了群聊",
        r"加入了群聊",
        r"退出群聊",
        r"群聊邀请确认",
        r"@所有人",
        r"系统消息",
    ]
    
    def __init__(self):
        self.system_pattern = re.compile("|".join(self.SYSTEM_PATTERNS))
    
    def parse(self, file_path: Path, target_sender: Optional[str] = None) -> ParsedData:
        """
        解析微信聊天记录
        
        Args:
            file_path: 聊天记录文件路径
            target_sender: 目标发送者名称（可选，用于过滤）
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            return self._parse_json(file_path, target_sender)
        elif file_path.suffix in ['.txt', '.csv']:
            return self._parse_txt(file_path, target_sender)
        elif file_path.suffix in ['.html', '.htm']:
            return self._parse_html(file_path, target_sender)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    def _parse_json(self, file_path: Path, target_sender: Optional[str]) -> ParsedData:
        """解析 JSON 格式"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        for item in data:
            # 跳过系统消息
            content = item.get('content', '')
            if self._is_system_message(content):
                continue
            
            sender = item.get('sender', item.get('nickName', 'Unknown'))
            
            # 如果指定了目标发送者，进行过滤
            if target_sender and sender != target_sender:
                continue
            
            msg = Message(
                timestamp=self._parse_timestamp(item.get('time', item.get('timestamp'))),
                sender=sender,
                content=content,
                message_type=item.get('type', 'text')
            )
            messages.append(msg)
        
        return ParsedData(
            source_type="wechat",
            messages=messages,
            metadata={
                "file_path": str(file_path),
                "total_messages": len(data),
                "filtered_messages": len(messages),
                "target_sender": target_sender
            }
        )
    
    def _parse_txt(self, file_path: Path, target_sender: Optional[str]) -> ParsedData:
        """解析 TXT 格式（常见导出格式）"""
        messages = []
        
        # 常见格式: 2023-01-01 12:00:00 发送者: 消息内容
        # 或: 发送者 2023/1/1 12:00 消息内容
        patterns = [
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\s+([^:]+):\s*(.+)',
            r'([^\d]+)\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2})\s*(.+)',
        ]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                for match in matches:
                    if len(match) == 3:
                        # 判断哪个是时间，哪个是发送者
                        if re.match(r'\d{4}', match[0]):
                            timestamp_str, sender, msg_content = match
                        else:
                            sender, timestamp_str, msg_content = match
                        
                        if self._is_system_message(msg_content):
                            continue
                        
                        if target_sender and sender.strip() != target_sender:
                            continue
                        
                        messages.append(Message(
                            timestamp=self._parse_timestamp(timestamp_str),
                            sender=sender.strip(),
                            content=msg_content.strip(),
                            message_type="text"
                        ))
                break
        
        return ParsedData(
            source_type="wechat",
            messages=messages,
            metadata={
                "file_path": str(file_path),
                "parsed_messages": len(messages),
                "target_sender": target_sender
            }
        )
    
    def _parse_html(self, file_path: Path, target_sender: Optional[str]) -> ParsedData:
        """解析 HTML 格式（WeChatMsg 等工具导出）"""
        from bs4 import BeautifulSoup
        
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        messages = []
        
        # 常见的 HTML 结构
        msg_divs = soup.find_all('div', class_=re.compile('message|msg|chat'))
        
        for div in msg_divs:
            # 提取发送者
            sender_elem = div.find(['span', 'div'], class_=re.compile('sender|nick|name'))
            sender = sender_elem.text.strip() if sender_elem else 'Unknown'
            
            # 提取内容
            content_elem = div.find(['span', 'div'], class_=re.compile('content|text|msg'))
            content = content_elem.text.strip() if content_elem else ''
            
            if self._is_system_message(content):
                continue
            
            if target_sender and sender != target_sender:
                continue
            
            # 提取时间
            time_elem = div.find(['span', 'div'], class_=re.compile('time|date'))
            timestamp = self._parse_timestamp(time_elem.text if time_elem else None)
            
            messages.append(Message(
                timestamp=timestamp,
                sender=sender,
                content=content,
                message_type="text"
            ))
        
        return ParsedData(
            source_type="wechat",
            messages=messages,
            metadata={
                "file_path": str(file_path),
                "parsed_messages": len(messages),
                "target_sender": target_sender
            }
        )
    
    def _is_system_message(self, content: str) -> bool:
        """判断是否为系统消息"""
        if not content:
            return True
        return bool(self.system_pattern.search(content))
    
    def _parse_timestamp(self, ts_str: Optional[str]) -> Optional[datetime]:
        """解析时间戳"""
        if not ts_str:
            return None
        
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M",
            "%Y年%m月%d日 %H:%M",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(str(ts_str).strip(), fmt)
            except ValueError:
                continue
        
        return None


class QQParser:
    """QQ 聊天记录解析器"""
    
    def parse(self, file_path: Path, target_sender: Optional[str] = None) -> ParsedData:
        """解析 QQ 聊天记录"""
        file_path = Path(file_path)
        
        if file_path.suffix == '.txt':
            return self._parse_txt(file_path, target_sender)
        elif file_path.suffix == '.mht':
            return self._parse_mht(file_path, target_sender)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    def _parse_txt(self, file_path: Path, target_sender: Optional[str]) -> ParsedData:
        """解析 QQ 导出的 txt 格式"""
        messages = []
        
        # QQ 格式: 2023-01-01 12:00:00 发送者(123456789)
        pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+([^\(]+)(?:\(\d+\))?\s*(.+)'
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for timestamp_str, sender, msg_content in matches:
            sender = sender.strip()
            
            if target_sender and sender != target_sender:
                continue
            
            messages.append(Message(
                timestamp=datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S"),
                sender=sender,
                content=msg_content.strip(),
                message_type="text"
            ))
        
        return ParsedData(
            source_type="qq",
            messages=messages,
            metadata={
                "file_path": str(file_path),
                "parsed_messages": len(messages),
                "target_sender": target_sender
            }
        )
    
    def _parse_mht(self, file_path: Path, target_sender: Optional[str]) -> ParsedData:
        """解析 QQ 导出的 mht 格式"""
        # MHT 解析较为复杂，这里简化处理
        # 实际使用时可以添加更完整的 MHT 解析逻辑
        raise NotImplementedError("MHT 格式解析暂未完成，请先转换为 TXT 格式")


class TextParser:
    """纯文本/日记解析器"""
    
    def parse(self, file_path: Path, title: Optional[str] = None) -> ParsedData:
        """
        解析文本文件
        
        支持格式：
        1. Markdown 格式（按标题分段）
        2. 纯文本（按空行分段）
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        messages = []
        
        # 尝试按 Markdown 标题分段
        sections = re.split(r'\n#{1,3}\s+', content)
        
        if len(sections) > 1:
            # Markdown 格式
            for section in sections[1:]:  # 跳过第一个空段
                lines = section.strip().split('\n')
                section_title = lines[0] if lines else "未命名"
                section_content = '\n'.join(lines[1:]).strip()
                
                if section_content:
                    messages.append(Message(
                        timestamp=None,
                        sender=title or "回忆",
                        content=f"【{section_title}】\n{section_content}",
                        message_type="memory"
                    ))
        else:
            # 纯文本格式，按空行分段
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                messages.append(Message(
                    timestamp=None,
                    sender=title or "回忆",
                    content=para,
                    message_type="memory"
                ))
        
        return ParsedData(
            source_type="text",
            messages=messages,
            metadata={
                "file_path": str(file_path),
                "title": title,
                "parsed_sections": len(messages)
            }
        )


class DataParser:
    """统一数据解析器入口"""
    
    def __init__(self):
        self.wechat_parser = WeChatParser()
        self.qq_parser = QQParser()
        self.text_parser = TextParser()
    
    def parse(self, file_path: Path, source_type: str, **kwargs) -> ParsedData:
        """
        统一解析入口
        
        Args:
            file_path: 文件路径
            source_type: 数据源类型 (wechat, qq, text, diary)
            **kwargs: 额外参数
                - target_sender: 目标发送者名称
                - title: 文本标题
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if source_type == "wechat":
            return self.wechat_parser.parse(file_path, kwargs.get('target_sender'))
        elif source_type == "qq":
            return self.qq_parser.parse(file_path, kwargs.get('target_sender'))
        elif source_type in ["text", "diary"]:
            return self.text_parser.parse(file_path, kwargs.get('title'))
        else:
            raise ValueError(f"未知的数据源类型: {source_type}")
    
    def batch_parse(self, file_configs: List[Dict]) -> List[ParsedData]:
        """
        批量解析多个文件
        
        Args:
            file_configs: 文件配置列表，每项包含:
                - file_path: 文件路径
                - source_type: 数据源类型
                - 其他参数
        """
        results = []
        for config in file_configs:
            try:
                result = self.parse(**config)
                results.append(result)
            except Exception as e:
                print(f"解析失败 {config.get('file_path')}: {e}")
        return results


# 便捷函数
def parse_wechat(file_path: str, target_sender: Optional[str] = None) -> ParsedData:
    """解析微信记录"""
    parser = WeChatParser()
    return parser.parse(Path(file_path), target_sender)


def parse_qq(file_path: str, target_sender: Optional[str] = None) -> ParsedData:
    """解析 QQ 记录"""
    parser = QQParser()
    return parser.parse(Path(file_path), target_sender)


def parse_text(file_path: str, title: Optional[str] = None) -> ParsedData:
    """解析文本文件"""
    parser = TextParser()
    return parser.parse(Path(file_path), title)
