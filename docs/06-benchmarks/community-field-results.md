# 同款 Thor 板社区实测经验汇总

> One-liner: Distilled field results from another owner of the identical
> p3960-0010 / Tegra264 / DriveOS 7.0.3 board — their llama.cpp patch stack
> hit 34.91 tok/s decode with FP8 + MTP K7 at 128K context.
>
> 日期：2026-09-07（用户提供资料，本仓库版：脱敏整理）

## 路线

魔改 llama.cpp/ggml 私有补丁体系（版本代号 A2/A4/A5），未用 vLLM/SGLang 官方栈（未安装成功）。两套模型方案：QUASAR（NVFP4 单一精度）和 Radix（NVFP4/FP8 混合）。

## 最好成绩

**Radix A4 = 34.91 tok/s 解码**（128K 上下文，输入 124739 tokens，Native FP8 路径，40GiB AI 池，FP16 KV，MTP K7）。

- FP8 原生路径比旧兼容路径：解码 +8.6%（32.14→34.91），AI 池峰值 34.27→27.57 GiB
- 冷首字 466 秒（128K 上下文 prefill 约 260-268 tok/s）
- MTP 接受率 94-99%，温度 68-69°C

## 其他配置成绩

| 配置 | tok/s | 备注 |
|---|---|---|
| A5-1 NEXTN/MTP + FP8 KV | 21.98 | |
| A5-2 DFlash2 blk8 | 17.18 | 比 A5-1 慢 21.83%，淘汰 |
| QUASAR A2 K7 | 37.1 | 64K 最快但有 6.7GiB BF16 扩展代价 |
| 某配置 | 8.522 | 业务质量不合格 |

## GPU 确认

7 TPC / 14 SM（官方 Thor-U SKU 规格，非车厂锁定），sm101/sm101a。MaxP 配置：1530MHz、2×NVENC、1×NVDEC、1×OFA、2×NVJPG、2×ISP。

对比 DGX Spark（GB10，6144 CUDA 核心）：Thor 算力小但媒体/视觉固定硬件丰富。官方 Thor 平台 H.265 编码 3.1Gpixel/s、H.264 约 3.0、解码 2.9/2.6。

## 存储实测

板载数据分区 105G（同款板已用 77G 剩 27G），另有地图分区 60G + 32G，合计可用 113G。17 个非零块设备 10 个挂载。

## 同款板主的结论（值得借鉴）

- Thor 是车载 Physical AI SoC，**不是小号 DGX Spark**
- 视频生成受 14 SM 限制
- ComfyUI 等消费级生态节点不会自动调用 OFA/PVA/NVENC
- 三阶段能力估计（相对 DGX Spark）：普通 ComfyUI 移植 = 50-65%，FP8/NVFP4 kernel 优化 = 65-85%，高度优化 + 媒体硬件卸载 = 80-100%

## 工程要点

CUDA 12.8 / AArch64 / SM101a / 动态库 ABI / FP8 / NVFP4 / Triton / GDN / 推测解码全部 PASS 验证可行。

⚠️ 编译注意：**exFAT 不区分大小写**——Linux 头文件有仅大小写不同的文件名（如 Windows.h/windows.h），源码包必须在 Linux 文件系统上解包。

工具链放私有目录，核实空间后再装，不动系统分区。
