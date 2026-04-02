#!/usr/bin/env python3
"""
微信聊天记录解析工具
支持多种导出格式
"""

import sys
import json
import argparse
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_parser import WeChatParser


def main():
    parser = argparse.ArgumentParser(description="解析微信聊天记录")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("-s", "--sender", help="目标发送者名称")
    parser.add_argument("-f", "--format", choices=["json", "txt"], default="json", help="输出格式")
    
    args = parser.parse_args()
    
    # 解析
    wechat_parser = WeChatParser()
    result = wechat_parser.parse(Path(args.input), args.sender)
    
    # 输出
    if args.format == "json":
        output = result.to_json()
    else:
        # 文本格式
        lines = [f"数据源: {result.source_type}", f"消息数: {len(result.messages)}", ""]
        for msg in result.messages:
            time_str = msg.timestamp.strftime("%Y-%m-%d %H:%M") if msg.timestamp else "未知时间"
            lines.append(f"[{time_str}] {msg.sender}: {msg.content}")
        output = "\n".join(lines)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"已保存到: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
