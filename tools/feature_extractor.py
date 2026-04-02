#!/usr/bin/env python3
"""
口语化特征提取工具
分析聊天记录，提取语气特征
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.persona_distiller import FeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="提取口语化特征")
    parser.add_argument("input", help="输入文件路径（JSON格式）")
    parser.add_argument("-o", "--output", help="输出文件路径")
    
    args = parser.parse_args()
    
    # 加载数据
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = [m['content'] for m in data.get('messages', [])]
    
    # 提取特征
    extractor = FeatureExtractor()
    features = extractor.extract_features(messages)
    
    # 格式化输出
    output = {
        "语气词频率": dict(features['particles']),
        "平均消息长度": features['avg_message_length'],
        "标点习惯": features['punctuation_style'],
        "常见句式": features['sentence_patterns'],
        "方言特征": features['dialect_hints'],
        "打招呼方式": features['greeting_style'],
        "结束语方式": features['ending_style'],
    }
    
    result = json.dumps(output, ensure_ascii=False, indent=2)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"已保存到: {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
