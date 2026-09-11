#!/usr/bin/env python3
"""
gpu_pool_check.py — 板端 GPU 大页池 / 统一内存余量检查

板上没有 nvidia-smi / nvtop，跑大模型前用它确认 GPU 池余量是否够。
用法: python3 gpu_pool_check.py [期望占用 GiB, 默认 20]

输出:
  - HugePages 池总量 / 空闲 / 已用
  - MemAvailable（真实余量，比 free 可靠）
  - 判定：给定期望占用是否放得下
"""
import sys

def read_meminfo():
    d = {}
    with open('/proc/meminfo') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                d[parts[0].rstrip(':')] = int(parts[1])  # kB
    return d

def main():
    want_gib = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    m = read_meminfo()

    hp_total = m.get('HugePages_Total', 0)
    hp_free = m.get('HugePages_Free', 0)
    hp_size = m.get('Hugepagesize', 2048)  # kB, 通常 2MB
    avail = m.get('MemAvailable', 0)       # kB

    pool_gib = hp_total * hp_size / 1024 / 1024
    free_gib = hp_free * hp_size / 1024 / 1024
    used_gib = pool_gib - free_gib
    avail_gib = avail / 1024 / 1024

    print(f'GPU 大页池:  总 {pool_gib:.1f}G | 空闲 {free_gib:.1f}G | 已用 {used_gib:.1f}G')
    print(f'系统可用:    MemAvailable {avail_gib:.1f}G')
    print()

    if free_gib >= want_gib:
        print(f'✅ 池内空闲 {free_gib:.1f}G ≥ 期望 {want_gib:.1f}G，放得下')
    else:
        print(f'❌ 池内空闲 {free_gib:.1f}G < 期望 {want_gib:.1f}G，放不下')
        print('   建议：扩池（见 docs/05-system-tuning/hugepage-pool.md）或 KV 量化')

if __name__ == '__main__':
    main()
