#!/usr/bin/env python3
"""
版本管理工具
用于备份和回滚纪念对象的数据
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def backup(slug: str, base_dir: Path) -> str:
    """
    备份当前版本
    
    Args:
        slug: 纪念对象标识
        base_dir: 基础目录
        
    Returns:
        备份版本号
    """
    reunion_dir = base_dir / slug
    versions_dir = reunion_dir / "versions"
    
    if not reunion_dir.exists():
        print(f"错误：找不到纪念对象 {slug}")
        return None
    
    versions_dir.mkdir(exist_ok=True)
    
    # 读取当前版本号
    meta_file = reunion_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        current_version = meta.get("version", "v1")
    else:
        current_version = "v1"
    
    # 生成备份目录名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{current_version}_{timestamp}"
    backup_dir = versions_dir / backup_name
    backup_dir.mkdir()
    
    # 备份文件
    files_to_backup = ["memory.md", "persona.md", "meta.json", "SKILL.md"]
    for filename in files_to_backup:
        src = reunion_dir / filename
        if src.exists():
            shutil.copy(src, backup_dir / filename)
    
    print(f"✓ 已备份到: {backup_dir}")
    return backup_name


def list_versions(slug: str, base_dir: Path) -> list:
    """
    列出所有版本
    
    Args:
        slug: 纪念对象标识
        base_dir: 基础目录
        
    Returns:
        版本列表
    """
    versions_dir = base_dir / slug / "versions"
    
    if not versions_dir.exists():
        print("没有历史版本")
        return []
    
    versions = sorted(versions_dir.iterdir(), reverse=True)
    
    print(f"纪念对象 {slug} 的历史版本：")
    for v in versions:
        if v.is_dir():
            # 获取备份时间
            meta_file = v / "meta.json"
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                version = meta.get("version", "?")
                updated = meta.get("updated_at", "?")
                print(f"  • {v.name} (version: {version}, updated: {updated})")
            else:
                print(f"  • {v.name}")
    
    return [v.name for v in versions if v.is_dir()]


def rollback(slug: str, version: str, base_dir: Path) -> bool:
    """
    回滚到指定版本
    
    Args:
        slug: 纪念对象标识
        version: 要回滚到的版本
        base_dir: 基础目录
        
    Returns:
        是否成功
    """
    reunion_dir = base_dir / slug
    version_dir = reunion_dir / "versions" / version
    
    if not version_dir.exists():
        print(f"错误：找不到版本 {version}")
        return False
    
    # 先备份当前版本
    print("先备份当前版本...")
    backup(slug, base_dir)
    
    # 恢复文件
    files_to_restore = ["memory.md", "persona.md", "meta.json", "SKILL.md"]
    for filename in files_to_restore:
        src = version_dir / filename
        dst = reunion_dir / filename
        if src.exists():
            shutil.copy(src, dst)
    
    print(f"✓ 已回滚到版本: {version}")
    return True


def main():
    parser = argparse.ArgumentParser(description="版本管理工具")
    parser.add_argument("--action", choices=["backup", "list", "rollback"], required=True)
    parser.add_argument("--slug", required=True, help="纪念对象标识")
    parser.add_argument("--base-dir", default="./reunions", help="基础目录")
    parser.add_argument("--version", help="要回滚到的版本（仅 rollback 需要）")
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    if args.action == "backup":
        backup(args.slug, base_dir)
    elif args.action == "list":
        list_versions(args.slug, base_dir)
    elif args.action == "rollback":
        if not args.version:
            print("错误：rollback 需要指定 --version")
            sys.exit(1)
        rollback(args.slug, args.version, base_dir)


if __name__ == "__main__":
    main()
