# Thor01 Qwen3.8-27B DFlash2 262K 生产部署包（T4/ARES 定案版）

> **预编译二进制 + 全套材料（ModelScope）**：
> 🔗 **https://www.modelscope.cn/models/navyyang/thor01-qwen38-27b-dflash2-t4ares-deploy**
>
> 2026-10-01 定稿。NVIDIA DRIVE Thor（p3960 / Tegra264 / sm_101a）上 llama.cpp 生产部署：
> **Qwen3.8-27B（NVFP4 FFN + F8 注意力）+ DFlash2 投机解码 + 多模态（mmproj），262144 上下文。**
> **200K decode 实测 22.46 / 21.88 / 22.11 t/s（三次）**；短提示 34.5–34.8 t/s。

## 下载什么

| 需求 | 去哪 |
|---|---|
| **直接跑**（推荐） | ModelScope 仓（链接见上）：`bin/llama-server-m2c-t4`（79 MiB 二进制）+ `bin/deploy.sh` 管理脚本 + 全件 SHA256SUMS |
| **自己编译** | ModelScope 仓 `src/`（工作树快照 / 补丁 / 工具链 / 构建脚本）或本仓本目录 `source/` |
| **部署 / 测试方法** | ModelScope 仓 `README.md`（115 行完整文档）或本仓本目录 `README.md` |
| **验证复现** | ModelScope 仓 `test/`：`bench8091.py` + 三次 200K 原始 jsonl |

下载后**务必先校验**：`sha256sum -c SHA256SUMS.txt`（13 件全过再跑）。

## 快速部署（三步）

```bash
# 1) 宿主机准备（每次开机一次；大页池必须冷启动分配——改池后需重启）
bash prebuilt-docs/02-launcher/prepare-host.sh   # 08 目录的脚本，46 GiB 池 + carveout
grep HugePages_Total /proc/meminfo               # 必须 23552
# 2) 模型落位 + 二进制就位（模型清单见完整 README §2）
./bin/deploy.sh start
# 3) 验证
curl --noproxy '*' http://<板IP>:8080/health     # {"status":"ok"}
```

启动日志四条生效判据（缺一即配置未生效）、模型 sha256、等效启动命令、测速方法、
已知边界（200K decode 30 t/s 暂无可证路径），见 ModelScope 仓 README 或本目录
[`README.md`](README.md) 完整文档。

## 实测数据（原始 jsonl 在 ModelScope 仓 `test/`）

| 口径 | decode t/s | prefill t/s | draft acceptance |
|---|---:|---:|---|
| 200K prompt + 384 out（09-28 冻结验收） | **22.4638** | ~133 | 0.51 |
| 200K 生产抽测（09-30） | **21.8846** | 134.3 | 0.5103 |
| 200K 复测（10-01） | **22.1094** | ~133 | mean_len 4.45 |
| 短提示 2K（10-01，三次连测） | **34.51 / 34.76 / 34.77** | 220.5 | 59.8% |

## 模型清单（3 件，不随仓分发，sha256 已核验）

| 文件 | 来源 |
|---|---|
| `RadixArk-F8attn-v2.gguf`（主模型，**内部量化件未公开发布**，sha `d15dac91…2e7e9`） | 基座 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，转换链见 [03-model-conversion](../03-model-conversion/README.md) |
| `Qwen3.8-27B-DFlash2-BF16.gguf`（投机草稿 1.9B） | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) |
| `mmproj-…-HauhauCS-Aggressive-BF16.gguf`（多模态投影） | HuggingFace 作者 HauhauCS |

## 相关

- 旧版 FP8 快路服务端（2026-09-19）走百度网盘，见 [08-fp8-fastpath-server](../08-fp8-fastpath-server/README.md)
- 本目录与 `tcgen05` 分支的关系：本目录 = 那条 T4 实验线的**定案发布版**（精度定论、与生产二进制逐字一致 sha `2066e681…`）
