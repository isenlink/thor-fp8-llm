# Thor 27B 模型首次上板战报

> One-liner: First successful on-board run of Qwen3.8-27B NVFP4 with native MTP
> speculative decoding via llama.cpp — 10.7 tok/s bare, ~13.5 tok/s with MTP,
> and the gap analysis to community-best 39+ tok/s.
>
> 日期：2026-09-08（本仓库版：脱敏整理）

## 🎯 核心成绩

| 配置 | pp128 | tg64/生成 | 备注 |
|---|---|---|---|
| Qwen3-4B Q4_K_M | 1516 t/s | **40.2 tok/s** | 基础参照 |
| Qwen3.8-27B NVFP4 裸跑 | 210 t/s | **10.7 tok/s** | 无投机解码 |
| **27B NVFP4 + MTP 投机解码** | - | **~13.5 tok/s** | **开箱即用** |

**MTP 数据**：draft acceptance 43-47%，mean draft len 2.3 —— 模型自带 MTP 头在 llama.cpp master 上直接可用（`--spec-type draft-mtp`），+25-30% 提速。

**距离社区 39.x tok/s 的差距**：还差 ~2.9 倍。缺口在 **tcgen05 手写 kernel**（prefill×2 + decode 提升）——当时为私有补丁，官方 master 没有，全网检索确认无公开 fork。

## 已验证的 27B 服务命令

```bash
./llama-server \
  -m Qwen3.8-27B-NVFP4-MTP-COMPACT-LOW.gguf \
  -ngl 99 -c 4096 --spec-type draft-mtp --port 8080
```
- OpenAI 兼容 API：`http://<board-ip>:8080/v1/chat/completions`
- 中文问答验证正常（量子计算/散文写作测试通过）

## 避坑清单

1. **跑大模型前必查 GPU 剩余**：板上无 nvidia-smi，需自制 alloc 测试小工具（total/free + 分档 alloc 测试）
2. **僵尸进程吃显存**：llama 进程 Ctrl-C 后可能残留，`pkill -f llama-server` 后再跑新的
3. NVFP4 GGUF 在 llama-bench 里显示为 "Q8_0"（NVFP4 数据存在 q8_0 块容器里，正常现象）
4. 板上没有 curl/strings，调试用 ssh 隧道回主机操作

## 模型档案

- **来源**：`esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF`（Hugging Face）
- **文件**：COMPACT-LOW 档 15.16GB，NVFP4 权重 + 内置 MTP(nextn) 头
- **其他档位**：MEDIUM 15.6G / HIGH 16.7G / HIGHEST 22.1G（超显存不可用）
- 官方底模：Qwen/Qwen3.8-27B（多模态）

## 后续优化路线（按性价比排序）

1. **MTP 参数调优**（当时就能做）：draft tokens 数、acceptance 阈值、不同温度下的接受率 → *已完成，见 NVFP4 优化实录*
2. **KV cache 量化**（`-ctk q8_0`）：省显存换更长上下文或更高 draft 长度 → *64K 上下文已上线*
3. **tcgen05 kernel**（最大缺口）：需社区补丁或参考官方 PR #17906（mxfp4 Blackwell）自研，工程量大
4. **dflash2**：server 已内置 `--spec-type draft-dflash` 选项，可直接对比测试
